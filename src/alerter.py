import json
import logging
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt_client
import torch
from kafka import KafkaConsumer

import model
from features import (
    WINDOW_SIZE,
    audio_record_to_power,
    build_signal_frame,
    contiguous_window_starts,
    csi_record_to_values,
    resample_audio,
    resample_csi,
    to_index,
)
from settings import (
    ALERT_COOLDOWN,
    KAFKA_BOOTSTRAP,
    MQTT_HOST,
    MQTT_PASS,
    MQTT_PORT,
    MQTT_TLS,
    MQTT_TOPIC,
    MQTT_USER,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_TOPIC = "audio.data"
CSI_TOPIC = "tcpdump.data"
STREAM_NAMES = {AUDIO_TOPIC: "오디오", CSI_TOPIC: "CSI"}
BUFFER_SECONDS = 3.0  # 윈도우(1.8초)보다 넉넉하게 최근 데이터만 들고 있는다
INFERENCE_INTERVAL = 0.1  # 학습 샘플 간격(stride 10스텝)과 같은 100ms마다 추론
# 윈도우 끝이 현재 시각보다 이만큼 이상 오래됐으면 추론하지 않는다 (fluent-bit flush 1초 + 여유)
MAX_LAG = 3.0
LOG_INTERVAL = 10.0  # 반복되는 문제는 이 간격으로 모아서 한 번만 로그를 남긴다


class SignalBuffer:
    """
    Kafka로 들어오는 오디오/CSI 레코드를 모아 두었다가 학습과 같은 방식
    (features.build_signal_frame)으로 10ms 격자 윈도우를 만든다.
    """

    def __init__(self, horizon: float = BUFFER_SECONDS, max_lag: float = MAX_LAG):
        self.horizon = horizon
        self.max_lag = max_lag
        self.streams: dict[str, deque[tuple[float, float | np.ndarray]]] = {
            AUDIO_TOPIC: deque(),
            CSI_TOPIC: deque(),
        }
        self.last_window_end: float | None = None
        self.skip_reason: str | None = None

    @property
    def audio(self):
        return self.streams[AUDIO_TOPIC]

    @property
    def csi(self):
        return self.streams[CSI_TOPIC]

    def add(self, topic: str, record: dict) -> str | None:
        """레코드를 버퍼에 넣는다. 넣지 못했으면 그 사유를, 성공하면 None을 돌려준다."""
        if topic not in self.streams:
            return f"알 수 없는 토픽 {topic}"
        ts = record.get("ts")
        if ts is None:
            return f"{STREAM_NAMES[topic]} 레코드에 ts 없음"
        if topic == AUDIO_TOPIC:
            value = audio_record_to_power(record)
            if value is None:
                return "오디오 레코드 형식 오류"
        else:
            value = csi_record_to_values(record)
            if value is None:
                return "CSI 서브캐리어 수 불일치 (ESP32 CSI 설정 확인)"
        self.streams[topic].append((float(ts), value))
        return None

    def prune(self):
        latest = max((buffer[-1][0] for buffer in self.streams.values() if buffer), default=None)
        if latest is None:
            return
        for buffer in self.streams.values():
            while buffer and buffer[0][0] < latest - self.horizon:
                buffer.popleft()

    def latest_window(self, now: float) -> np.ndarray | None:
        """
        두 스트림이 모두 도착한 가장 최근 1.8초 윈도우 (채널, 시간).

        추론할 수 없으면 None을 돌려주고, 문제가 있는 경우 그 사유를 skip_reason에 남긴다.
        이미 추론한 구간이거나(새 데이터 없음) 윈도우가 오래된 경우에도 None이다.

        :param now: 현재 epoch 초. 센서 레코드의 ts와 같은 시계여야 한다.
        """
        self.prune()
        self.skip_reason = None
        for topic, buffer in self.streams.items():
            if not buffer or buffer[-1][0] < now - self.max_lag:
                self.skip_reason = f"{STREAM_NAMES[topic]} 데이터가 {self.max_lag:g}초 넘게 없음"
                return None

        # 늦게 도착하는 스트림 쪽을 보간으로 채우지 않도록 두 스트림이 모두 있는 시점까지만 쓴다
        end_ts = min(self.audio[-1][0], self.csi[-1][0])
        if self.last_window_end is not None and end_ts <= self.last_window_end:
            return None

        frame = build_signal_frame(
            resample_audio(*zip(*self.audio)),
            resample_csi(*zip(*self.csi)),
        )
        window = frame.loc[: to_index([end_ts])[0]].iloc[-WINDOW_SIZE:]
        if len(contiguous_window_starts(window.index, WINDOW_SIZE, stride=1)) == 0:
            self.skip_reason = "연속된 1.8초 구간이 아직 없음"
            return None
        self.last_window_end = end_ts
        return window.to_numpy(dtype=np.float32).T.copy()


class ProblemLog:
    """같은 문제가 초당 수십 번 반복돼도 interval마다 건수를 모아 한 줄로 남긴다."""

    def __init__(self, interval: float = LOG_INTERVAL):
        self.interval = interval
        self.counts: dict[str, int] = {}
        self.last_flush = time.monotonic()

    def count(self, reason: str):
        self.counts[reason] = self.counts.get(reason, 0) + 1

    def flush(self, now: float) -> list[str]:
        if now - self.last_flush < self.interval:
            return []
        self.last_flush = now
        lines = [f"{reason} ({n}회/{self.interval:g}초)" for reason, n in self.counts.items()]
        for line in lines:
            logger.warning(line)
        self.counts.clear()
        return lines


class AlertThrottle:
    """낙상이 연속으로 감지되어도 cooldown 초에 한 번만 알림을 보낸다."""

    def __init__(self, cooldown: float = ALERT_COOLDOWN):
        self.cooldown = cooldown
        self.last_alert: float | None = None

    def should_alert(self, now: float) -> bool:
        if self.last_alert is not None and now - self.last_alert < self.cooldown:
            return False
        self.last_alert = now
        return True


def connect_mqtt() -> mqtt_client.Client:
    client = mqtt_client.Client(
        mqtt_client.CallbackAPIVersion.VERSION2,
        client_id="alerter",
        protocol=mqtt_client.MQTTv5,
    )
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)
    if MQTT_TLS:
        client.tls_set()
    client.connect(MQTT_HOST, port=MQTT_PORT)
    client.loop_start()
    return client


def main():
    classifier = model.Net()
    classifier.load_model()
    classifier.eval()

    # 컨슈머 그룹/오프셋 커밋 없이 항상 최신 레코드부터 읽는다.
    # 그룹 오프셋을 이어 읽으면 재시작 시 멈춰 있던 동안의 데이터를 처리해 지난 낙상에 알림이 간다.
    consumer = KafkaConsumer(
        AUDIO_TOPIC,
        CSI_TOPIC,
        group_id=None,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )

    alerter = connect_mqtt()
    throttle = AlertThrottle()
    buffer = SignalBuffer()
    problems = ProblemLog()
    last_inference = 0.0

    while True:
        for partition, messages in consumer.poll(timeout_ms=100).items():
            for message in messages:
                try:
                    record = json.loads(message.value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    problems.count(f"{partition.topic} JSON 파싱 오류")
                    continue
                reason = buffer.add(partition.topic, record)
                if reason is not None:
                    problems.count(reason)

        problems.flush(time.monotonic())
        if time.monotonic() - last_inference < INFERENCE_INTERVAL:
            continue
        last_inference = time.monotonic()

        window = buffer.latest_window(time.time())
        if window is None:
            if buffer.skip_reason is not None:
                problems.count(f"추론 건너뜀: {buffer.skip_reason}")
            continue
        try:
            predicted = classifier.predict(torch.from_numpy(window).unsqueeze(0))
        except Exception as e:
            problems.count(f"추론 오류: {e!r}")
            continue
        if predicted.item() == 1 and throttle.should_alert(time.monotonic()):
            alerter.publish(MQTT_TOPIC, b"alert", qos=1)
            logger.info("낙상 감지 알림 전송")


if __name__ == "__main__":
    main()
