import time

import numpy as np
import sounddevice as sd
from fluent import asyncsender as sender

SAMPLE_RATE = 44100  # 샘플링 레이트
BLOCK_DURATION = 1.0  # 블록 단위 시간 (초)
FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트

logger = sender.FluentSender(
    "audio", host=FLUENT_BIT_HOST, port=FLUENT_BIT_PORT, nanosecond_precision=True, queue_maxsize=1000,
    queue_circular=True
)


def send_data_to_fluent_bit(data):
    for item in data:
        timestamp, value = item
        logger.emit_with_time("data", timestamp, {"data": float(value)})


def record_audio(duration, sample_rate):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio_data


def process_audio_with_timestamps(audio_data, sample_rate):
    num_samples = audio_data.shape[0]
    timestamps = np.linspace(0, num_samples / sample_rate, num_samples)
    current_time = time.time()

    timestamped_data = []
    for i in range(num_samples):
        timestamped_data.append((current_time + timestamps[i], audio_data[i][0]))

    return timestamped_data


def record_and_process_audio_continuous(callback, sample_rate, block_duration):
    def callback_wrapper(indata, _frames, _time, _status):
        timestamped_data = process_audio_with_timestamps(indata, sample_rate)
        callback(timestamped_data)

    block_size = int(sample_rate * block_duration)
    with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            blocksize=block_size,
            callback=callback_wrapper,
    ):
        while True:
            sd.sleep(int(block_duration * 1000))


def send_data_callback(timestamped_data):
    send_data_to_fluent_bit(timestamped_data)


record_and_process_audio_continuous(send_data_callback, SAMPLE_RATE, BLOCK_DURATION)
