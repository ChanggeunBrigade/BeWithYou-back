import time

import numpy as np
import sounddevice as sd
from fluent import asyncsender as sender

SAMPLE_RATE = 44100  # 샘플링 레이트
BLOCK_DURATION = 0.1  # 사운드카드 콜백 단위 시간 (초)
FEATURE_DURATION = 0.01  # RMS를 계산해 전송하는 단위 시간 (초), features.RESAMPLE_RATE와 같음
FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트

logger = sender.FluentSender(
    "audio",
    host=FLUENT_BIT_HOST,
    port=FLUENT_BIT_PORT,
    nanosecond_precision=True,
    queue_maxsize=1000,
    queue_circular=True,
)


def block_rms(samples: np.ndarray, sample_rate: int, feature_duration: float) -> np.ndarray:
    """샘플을 feature_duration 단위로 나눠 각 구간의 RMS를 구한다."""
    size = int(sample_rate * feature_duration)
    usable = len(samples) - len(samples) % size
    chunks = samples[:usable].reshape(-1, size)
    return np.sqrt(np.mean(np.square(chunks, dtype=np.float64), axis=1))


def send_block(indata: np.ndarray, block_end: float):
    rms = block_rms(indata[:, 0], SAMPLE_RATE, FEATURE_DURATION)
    block_start = block_end - len(rms) * FEATURE_DURATION
    for i, value in enumerate(rms):
        # 파형 샘플 하나하나(초당 44100건) 대신 10ms 구간의 RMS만 보낸다 (초당 100건)
        logger.emit_with_time("data", block_start + i * FEATURE_DURATION, {"rms": float(value)})


def record_and_send_continuous():
    def callback(indata, _frames, _time, _status):
        send_block(indata, time.time())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
        callback=callback,
    ):
        while True:
            sd.sleep(1000)


if __name__ == "__main__":
    record_and_send_continuous()
