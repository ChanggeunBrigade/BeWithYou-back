import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import database
from features import (
    FEATURE_COLUMNS,
    RESAMPLE_RATE,
    WINDOW_SIZE,
    build_signal_frame,
    contiguous_window_starts,
    csi_record_to_values,
    resample_audio,
    resample_csi,
    to_index,
)

# 윈도우(180스텝) 안에서 낙상 라벨이 이 스텝 수를 넘으면 낙상 윈도우로 본다 (0.3초)
LABEL_THRESHOLD = 30
# 라벨을 앞뒤로 이어 붙이는 최대 스텝 수 (5초). 프레임마다 라벨을 달지 않아도 되게 한다
LABEL_FILL = pd.Timedelta(RESAMPLE_RATE) * 500


def build_audio_frame(power_rows: list[tuple[int, float]], step: float) -> pd.DataFrame:
    """Database.get_audio_power 결과(격자 번호, 평균 전력)로 오디오 RMS 채널을 만든다."""
    buckets = np.array([row[0] for row in power_rows], dtype=np.float64)
    # 격자 중앙 시각을 써서 부동소수점 오차로 이웃 격자에 들어가는 일이 없게 한다
    return resample_audio((buckets + 0.5) * step, [row[1] for row in power_rows])


def build_csi_frame(csi_rows) -> pd.DataFrame:
    """csi_rows: (ts, amplitudes, phases) 이터레이터"""
    ts, values = [], []
    for row_ts, amplitudes, phases in csi_rows:
        parsed = csi_record_to_values({"amplitudes": amplitudes, "phases": phases})
        if parsed is None:
            continue
        ts.append(row_ts)
        values.append(parsed)
    return resample_csi(ts, values)


def label_signal(index: pd.DatetimeIndex, label_rows: list[tuple[float, int]]) -> pd.Series:
    """
    신호 격자의 각 시점에 라벨을 붙인다. 가장 최근 라벨(LABEL_FILL 이내)을 쓰고, 없으면
    다음 라벨(LABEL_FILL 이내)을 쓰며, 둘 다 없으면 NaN이다.

    :param label_rows: Database.get_labels 결과 (epoch 초, 라벨)
    """
    labels = pd.DataFrame(
        {
            "time": to_index([row[0] for row in label_rows]),
            "label": np.asarray([row[1] for row in label_rows], dtype=np.float32),
        }
    ).sort_values("time")
    grid = pd.DataFrame({"time": index})
    # 0/1 분류 데이터이므로 interpolate나 mean 대신 앞/뒤 라벨만 이어 붙인다
    merged = [
        pd.merge_asof(grid, labels, on="time", direction=direction, tolerance=LABEL_FILL)
        for direction in ("backward", "forward")
    ]
    result = merged[0]["label"].fillna(merged[1]["label"])
    result.index = index
    return result


class TrainDataset(Dataset):
    def __init__(self, start=None, end=None, stride: int = 10):
        """
        :param start: 학습에 쓸 기간 시작 (DB_TIMEZONE 기준, 예: "2024-05-01 09:00").
            None이면 처음부터
        :param end: 학습에 쓸 기간 끝 (포함하지 않음). None이면 끝까지
        :param stride: 윈도우 시작 간격(스텝). 기본 10스텝(100ms)마다 하나의 샘플을 만든다.
        """
        self.db = database.Database()
        step = pd.Timedelta(RESAMPLE_RATE).total_seconds()

        signal = build_signal_frame(
            build_audio_frame(self.db.get_audio_power(step, start, end), step),
            build_csi_frame(self.db.iter_csi(start, end)),
        )
        labels = label_signal(signal.index, self.db.get_labels(start, end))
        data = signal.assign(label=labels).dropna()
        self.index = data.index

        # 윈도우를 미리 잘라 두지 않고 전체 시계열 하나만 들고 있다가 필요할 때 잘라 쓴다
        self.x = torch.from_numpy(data[FEATURE_COLUMNS].to_numpy(dtype=np.float32).T.copy())
        labels = data["label"].to_numpy(dtype=np.float32)
        self.starts = contiguous_window_starts(data.index, WINDOW_SIZE, stride)

        label_cumsum = np.concatenate([[0.0], np.cumsum(labels)])
        label_sum = label_cumsum[self.starts + WINDOW_SIZE] - label_cumsum[self.starts]
        self.y = (label_sum > LABEL_THRESHOLD).astype(np.float32)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        start = self.starts[item]
        return self.x[:, start : start + WINDOW_SIZE], self.y[item]


if __name__ == "__main__":
    dataset = TrainDataset()
    print(f"samples: {len(dataset)}, positive: {int(dataset.y.sum())}")
