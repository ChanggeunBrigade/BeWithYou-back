import numpy as np

from alerter import AUDIO_TOPIC, CSI_TOPIC, AlertThrottle, ProblemLog, SignalBuffer
from features import N_FEATURES, N_SUBCARRIERS, WINDOW_SIZE

START = 1_700_000_000.0


def fill(buffer, seconds, csi_until=None, start=START):
    for i in range(int(seconds * 100)):
        ts = start + i / 100
        assert buffer.add(AUDIO_TOPIC, {"ts": ts, "rms": 0.2}) is None
        if i % 2 == 0 and (csi_until is None or ts <= start + csi_until):
            amplitudes = [float(i)] * N_SUBCARRIERS
            record = {"ts": ts, "amplitudes": amplitudes, "phases": [0.5] * N_SUBCARRIERS}
            assert buffer.add(CSI_TOPIC, record) is None


def test_window_matches_model_input():
    buffer = SignalBuffer()
    fill(buffer, seconds=2.5)
    window = buffer.latest_window(now=START + 2.5)
    assert window.shape == (N_FEATURES, WINDOW_SIZE)
    assert np.allclose(window[0], 0.2)  # 오디오 RMS 채널이 맨 앞
    assert np.allclose(window[1 + N_SUBCARRIERS :], 0.5)


def test_not_enough_data():
    buffer = SignalBuffer()
    fill(buffer, seconds=1.0)
    assert buffer.latest_window(now=START + 1.0) is None
    assert "연속된" in buffer.skip_reason


def test_window_ends_where_both_streams_exist():
    buffer = SignalBuffer()
    fill(buffer, seconds=3.0, csi_until=2.0)
    window = buffer.latest_window(now=START + 3.0)
    # CSI가 끊긴 뒤의 구간을 마지막 값으로 채우지 않는다
    assert window[1, -1] == 200.0


def test_stale_data_is_not_inferred():
    buffer = SignalBuffer(max_lag=3.0)
    fill(buffer, seconds=2.5)
    # 센서가 멈춘 뒤(또는 재시작 후 지난 데이터)에는 추론하지 않는다
    assert buffer.latest_window(now=START + 60) is None
    assert "초 넘게 없음" in buffer.skip_reason


def test_one_stream_stopped():
    buffer = SignalBuffer(max_lag=1.0)
    fill(buffer, seconds=5.0, csi_until=2.0)
    assert buffer.latest_window(now=START + 5.0) is None
    assert buffer.skip_reason.startswith("CSI")


def test_same_window_is_not_inferred_twice():
    buffer = SignalBuffer()
    fill(buffer, seconds=2.5)
    assert buffer.latest_window(now=START + 2.5) is not None
    assert buffer.latest_window(now=START + 2.6) is None
    assert buffer.skip_reason is None  # 문제 상황이 아니라 새 데이터를 기다리는 것
    fill(buffer, seconds=0.1, start=START + 2.5)
    assert buffer.latest_window(now=START + 2.6) is not None


def test_rejects_malformed_records():
    buffer = SignalBuffer()
    assert buffer.add(AUDIO_TOPIC, {"rms": 0.1}) is not None  # ts 없음
    bad_csi = {"ts": START, "amplitudes": [1.0] * 10, "phases": [0.0] * 10}
    assert "서브캐리어" in buffer.add(CSI_TOPIC, bad_csi)
    assert buffer.add("unknown", {"ts": START}) is not None


def test_prune_keeps_recent_data_only():
    buffer = SignalBuffer(horizon=1.0)
    fill(buffer, seconds=5.0)
    buffer.prune()
    assert buffer.audio[-1][0] - buffer.audio[0][0] <= 1.0


def test_problem_log_aggregates():
    problems = ProblemLog(interval=10)
    for _ in range(50):
        problems.count("CSI 서브캐리어 수 불일치")
    assert problems.flush(problems.last_flush + 1) == []
    assert problems.flush(problems.last_flush + 10) == ["CSI 서브캐리어 수 불일치 (50회/10초)"]
    assert problems.flush(problems.last_flush + 10) == []


def test_alert_throttle():
    throttle = AlertThrottle(cooldown=30)
    assert throttle.should_alert(100.0)
    assert not throttle.should_alert(110.0)
    assert throttle.should_alert(131.0)
