import numpy as np

from alerter import AUDIO_TOPIC, CSI_TOPIC, SignalBuffer
from features import N_FEATURES, N_SUBCARRIERS, WINDOW_SIZE

START = 1_700_000_000.0


def fill(buffer, seconds, csi_until=None):
    for i in range(int(seconds * 100)):
        ts = START + i / 100
        buffer.add(AUDIO_TOPIC, {"ts": ts, "rms": 0.2})
        if i % 2 == 0 and (csi_until is None or ts <= START + csi_until):
            amplitudes = [float(i)] * N_SUBCARRIERS
            buffer.add(CSI_TOPIC, {"ts": ts, "amplitudes": amplitudes, "phases": [0.5] * 64})


def test_window_matches_model_input():
    buffer = SignalBuffer()
    fill(buffer, seconds=2.5)
    window = buffer.latest_window()
    assert window.shape == (N_FEATURES, WINDOW_SIZE)
    assert np.allclose(window[0], 0.2)  # 오디오 RMS 채널이 맨 앞
    assert np.allclose(window[1 + N_SUBCARRIERS :], 0.5)


def test_not_enough_data():
    buffer = SignalBuffer()
    fill(buffer, seconds=1.0)
    assert buffer.latest_window() is None


def test_window_ends_where_both_streams_exist():
    buffer = SignalBuffer()
    fill(buffer, seconds=3.0, csi_until=2.0)
    window = buffer.latest_window()
    # CSI가 끊긴 뒤의 구간을 마지막 값으로 채우지 않는다
    assert window[1, -1] == 200.0


def test_rejects_malformed_records():
    buffer = SignalBuffer()
    assert not buffer.add(AUDIO_TOPIC, {"rms": 0.1})  # ts 없음
    assert not buffer.add(CSI_TOPIC, {"ts": START, "amplitudes": [1.0] * 10, "phases": [0.0] * 10})
    assert not buffer.add("unknown", {"ts": START})


def test_prune_keeps_recent_data_only():
    buffer = SignalBuffer(horizon=1.0)
    fill(buffer, seconds=5.0)
    buffer.prune()
    assert buffer.audio[-1][0] - buffer.audio[0][0] <= 1.0
