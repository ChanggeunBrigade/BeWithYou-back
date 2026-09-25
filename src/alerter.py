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
    csi_record_to_row,
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
BUFFER_SECONDS = 3.0  # 윈도우(1.8초)보다 넉넉하게 최근 데이터만 들고 있는다
INFERENCE_INTERVAL = 0.1  # 학습 샘플 간격(stride 10스텝)과 같은 100ms마다 추론


class SignalBuffer:
    """
    Kafka로 들어오는 오디오/CSI 레코드를 모아 두었다가 학습과 같은 방식
    (features.build_signal_frame)으로 10ms 격자 윈도우를 만든다.
    """

    def __init__(self, horizon: float = BUFFER_SECONDS):
        self.horizon = horizon
        self.audio: deque[tuple[float, float]] = deque()
        self.csi: deque[tuple[float, dict]] = deque()

    def add(self, topic: str, record: dict) -> bool:
        ts = record.get("ts")
        if ts is None:
            return False
        if topic == AUDIO_TOPIC:
            power = audio_record_to_power(record)
            if power is None:
                return False
            self.audio.append((float(ts), power))
        elif topic == CSI_TOPIC:
            row = csi_record_to_row(record)
            if row is None:
                return False
            self.csi.append((float(ts), row))
        else:
            return False
        return True

    def prune(self):
        if not self.audio or not self.csi:
            return
        oldest = max(self.audio[-1][0], self.csi[-1][0]) - self.horizon
        for buffer in (self.audio, self.csi):
            while buffer and buffer[0][0] < oldest:
                buffer.popleft()

    def latest_window(self) -> np.ndarray | None:
        """두 스트림이 모두 도착한 가장 최근 1.8초 윈도우 (채널, 시간). 없으면 None."""
        self.prune()
        if not self.audio or not self.csi:
            return None
        frame = build_signal_frame(
            resample_audio(*zip(*self.audio)),
            resample_csi(*zip(*self.csi)),
        )
        # 늦게 도착하는 스트림 쪽을 보간으로 채우지 않도록 두 스트림이 모두 있는 시점까지만 쓴다
        end = to_index([min(self.audio[-1][0], self.csi[-1][0])])[0]
        window = frame.loc[:end].iloc[-WINDOW_SIZE:]
        if len(contiguous_window_starts(window.index, WINDOW_SIZE, stride=1)) == 0:
            return None
        return window.to_numpy(dtype=np.float32).T.copy()


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

    consumer = KafkaConsumer(
        AUDIO_TOPIC,
        CSI_TOPIC,
        group_id="alert",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        auto_offset_reset="latest",
    )

    alerter = connect_mqtt()
    throttle = AlertThrottle()
    buffer = SignalBuffer()
    last_inference = 0.0

    while True:
        for partition, messages in consumer.poll(timeout_ms=100).items():
            for message in messages:
                try:
                    record = json.loads(message.value.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("데이터 파싱 오류: %s", e)
                    continue
                if not buffer.add(partition.topic, record):
                    logger.warning("형식이 맞지 않는 레코드를 건너뜀: %s", partition.topic)

        if time.monotonic() - last_inference < INFERENCE_INTERVAL:
            continue
        last_inference = time.monotonic()

        window = buffer.latest_window()
        if window is None:
            continue
        try:
            predicted = classifier.predict(torch.from_numpy(window).unsqueeze(0))
        except Exception as e:
            logger.error("추론 오류: %s", e)
            continue
        if predicted.item() == 1 and throttle.should_alert(time.monotonic()):
            alerter.publish(MQTT_TOPIC, b"alert", qos=1)
            logger.info("낙상 감지 알림 전송")


if __name__ == "__main__":
    main()
