import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import database
from features import (
    FEATURE_COLUMNS,
    INTERPOLATE_LIMIT,
    RESAMPLE_RATE,
    WINDOW_SIZE,
    audio_record_to_power,
    build_signal_frame,
    contiguous_window_starts,
    csi_record_to_row,
    resample_audio,
    resample_csi,
)
from settings import DB_TIMEZONE

# 윈도우(180스텝) 안에서 낙상 라벨이 이 스텝 수를 넘으면 낙상 윈도우로 본다 (0.3초)
LABEL_THRESHOLD = 30


def build_audio_frame(audio_rows: list) -> pd.DataFrame:
    ts, power = [], []
    for row in audio_rows:
        value = audio_record_to_power(row[2])
        if value is None:
            continue
        ts.append(row[2]["ts"])
        power.append(value)
    return resample_audio(ts, power)


def build_csi_frame(csi_rows: list) -> pd.DataFrame:
    ts, rows = [], []
    for row in csi_rows:
        parsed = csi_record_to_row(row[2])
        if parsed is None:
            continue
        ts.append(row[2]["ts"])
        rows.append(parsed)
    return resample_csi(ts, rows)


def to_utc_naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """DB 시각을 신호 데이터와 같은 UTC 기준 naive datetime으로 맞춘다."""
    if index.tz is None:
        index = index.tz_localize(DB_TIMEZONE)
    return index.tz_convert("UTC").tz_localize(None)


def build_label_frame(label_rows: list) -> pd.DataFrame:
    label = pd.DataFrame(label_rows, columns=["time", "label"])
    label.index = to_utc_naive(pd.DatetimeIndex(label.pop("time")))
    # 0/1 분류 데이터이므로 interpolate나 mean 대신 ffill/bfill만 사용한다
    return (
        label.sort_index()
        .resample(RESAMPLE_RATE)
        .ffill(limit=INTERPOLATE_LIMIT)
        .bfill(limit=INTERPOLATE_LIMIT)
    )


class TrainDataset(Dataset):
    def __init__(self, stride: int = 10):
        """
        :param stride: 윈도우 시작 간격(스텝). 기본 10스텝(100ms)마다 하나의 샘플을 만든다.
        """
        self.db = database.Database()

        signal = build_signal_frame(
            build_audio_frame(self.db.get_table_data("audio")),
            build_csi_frame(self.db.get_table_data("tcpdump")),
        )
        data = signal.join(build_label_frame(self.db.get_table_data("label"))).dropna()
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
