"""
학습(dataload.py)과 실시간 추론(alerter.py)이 공유하는 입력 특징 정의.

두 경로가 반드시 같은 격자 간격, 같은 채널 순서로 텐서를 만들도록 여기서만 정의한다.
모든 시각은 epoch 초(fluent-bit가 붙이는 `ts`)를 UTC 기준 naive datetime으로 바꿔 쓴다.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

N_SUBCARRIERS = 64
RESAMPLE_RATE = "10ms"  # 모델 입력 한 스텝의 간격
WINDOW_SIZE = 180  # 10ms * 180 = 1.8초
# 센서 결측을 직선 보간으로 메우는 최대 스텝 수 (200ms).
# 이보다 긴 결측은 지어내지 않고 비워 두며, 그 구간에 걸친 윈도우는 쓰지 않는다.
INTERPOLATE_LIMIT = 20
STEP_NS = pd.Timedelta(RESAMPLE_RATE).value

AMPLITUDE_COLUMNS = [f"amplitudes_{sub}" for sub in range(N_SUBCARRIERS)]
PHASE_COLUMNS = [f"phases_{sub}" for sub in range(N_SUBCARRIERS)]
CSI_COLUMNS = AMPLITUDE_COLUMNS + PHASE_COLUMNS

# 모델 입력 채널 순서: 오디오 1채널 + 진폭 64채널 + 위상 64채널 = 129채널
FEATURE_COLUMNS = ["audio"] + CSI_COLUMNS
N_FEATURES = len(FEATURE_COLUMNS)


def csi_record_to_values(record: dict) -> np.ndarray | None:
    """
    esp.py가 보낸 CSI 레코드를 CSI_COLUMNS 순서(진폭 64 + 위상 64)의 배열로 만든다.
    서브캐리어 수가 맞지 않으면 None.
    """
    amplitudes = record.get("amplitudes")
    phases = record.get("phases")
    if (
        amplitudes is None
        or phases is None
        or len(amplitudes) != N_SUBCARRIERS
        or len(phases) != N_SUBCARRIERS
    ):
        return None
    return np.asarray([*amplitudes, *phases], dtype=np.float32)


def audio_record_to_power(record: dict) -> float | None:
    """
    오디오 레코드의 평균 전력(진폭 제곱).

    audio.py는 10ms 블록의 RMS를 `rms`로 보낸다. 예전 데이터는 샘플 하나를 `data`로
    보냈으므로 그 제곱을 쓰면 리샘플링 후 같은 RMS 값이 된다.
    """
    if "rms" in record:
        return float(record["rms"]) ** 2
    if "data" in record:
        return float(record["data"]) ** 2
    return None


def to_index(ts: Sequence[float]) -> pd.DatetimeIndex:
    # pandas 버전에 따라 해상도(us/ns)가 달라지지 않도록 ns로 고정한다 (asi8 // STEP_NS 계산)
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(ts, dtype=np.float64), unit="s"))
    return index.as_unit("ns")


def _fill_grid(bins: np.ndarray, limit: int) -> np.ndarray:
    """
    데이터가 있는 격자 번호(정렬, 중복 없음)에 limit 스텝 이하인 빈칸만 채운 격자.

    전체 기간을 10ms 격자로 펼치지 않으므로 데이터 사이에 긴 공백이 있어도
    메모리 사용량은 실제 데이터 양에만 비례한다.
    """
    if len(bins) < 2:
        return bins
    gaps = np.diff(bins)
    fill = (gaps > 1) & (gaps <= limit + 1)
    extra = [np.arange(a + 1, b) for a, b in zip(bins[:-1][fill], bins[1:][fill])]
    return np.sort(np.concatenate([bins, *extra])) if extra else bins


def _to_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """레코드를 10ms 격자별로 평균내고 짧은 결측만 보간한다 (긴 결측은 행이 없다)."""
    bins = frame.index.asi8 // STEP_NS
    gridded = frame.groupby(bins).mean()
    grid = _fill_grid(gridded.index.to_numpy(), INTERPOLATE_LIMIT)
    gridded = gridded.reindex(grid).interpolate(limit_area="inside")
    gridded.index = pd.DatetimeIndex(pd.to_datetime(grid * STEP_NS, unit="ns")).as_unit("ns")
    return gridded


def resample_audio(ts: Sequence[float], power: Sequence[float]) -> pd.DataFrame:
    """오디오 전력을 격자별로 평균낸 뒤 제곱근을 취해 RMS 채널을 만든다."""
    frame = pd.DataFrame({"audio": np.asarray(power, dtype=np.float32)}, index=to_index(ts))
    return np.sqrt(_to_grid(frame))


def resample_csi(ts: Sequence[float], values: Sequence[np.ndarray]) -> pd.DataFrame:
    """values: 레코드마다 csi_record_to_values 결과 (또는 (N, 128) 배열)"""
    values = np.asarray(values, dtype=np.float32).reshape(-1, len(CSI_COLUMNS))
    return _to_grid(pd.DataFrame(values, columns=CSI_COLUMNS, index=to_index(ts)))


def build_signal_frame(audio: pd.DataFrame, csi: pd.DataFrame) -> pd.DataFrame:
    """
    오디오와 CSI 격자를 합쳐 모델 채널 순서의 프레임을 만든다.
    두 스트림이 모두 있는 격자만 남긴다 (한쪽이 비는 구간을 보간으로 지어내지 않음).
    """
    return audio.join(csi, how="inner").dropna()[FEATURE_COLUMNS]


def contiguous_window_starts(index: pd.DatetimeIndex, window: int, stride: int) -> np.ndarray:
    """결측 구간을 건너뛰지 않고 window 스텝이 연속으로 이어지는 윈도우의 시작 위치."""
    if len(index) < window:
        return np.array([], dtype=np.int64)
    span = index[window - 1 :] - index[: len(index) - window + 1]
    ok = np.asarray(span == pd.Timedelta(RESAMPLE_RATE) * (window - 1))
    starts = np.flatnonzero(ok)
    return starts[starts % stride == 0]
