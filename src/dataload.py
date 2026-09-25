import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import database
from features import (
    CSI_COLUMNS,
    FEATURE_COLUMNS,
    INTERPOLATE_LIMIT,
    RESAMPLE_RATE,
    WINDOW_SIZE,
    csi_record_to_row,
)

# 윈도우(180스텝) 안에서 낙상 라벨이 이 스텝 수를 넘으면 낙상 윈도우로 본다 (0.3초)
LABEL_THRESHOLD = 30


def to_signal_index(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(ts, unit="s") + pd.Timedelta(hours=9))


def build_audio_frame(audio_rows: list) -> pd.DataFrame:
    records = [row[2] for row in audio_rows]
    audio = pd.DataFrame(
        {"audio": [r["data"] for r in records]},
        index=to_signal_index(pd.Series([r["ts"] for r in records])),
    )
    return audio.sort_index().resample(RESAMPLE_RATE).mean().interpolate(limit=INTERPOLATE_LIMIT)


def build_csi_frame(csi_rows: list) -> pd.DataFrame:
    rows, ts = [], []
    for row in csi_rows:
        parsed = csi_record_to_row(row[2])
        if parsed is None:
            continue
        rows.append(parsed)
        ts.append(row[2]["ts"])
    csi = pd.DataFrame(rows, columns=CSI_COLUMNS, index=to_signal_index(pd.Series(ts)))
    return csi.sort_index().resample(RESAMPLE_RATE).mean().interpolate(limit=INTERPOLATE_LIMIT)


def build_label_frame(label_rows: list) -> pd.DataFrame:
    # 0/1 분류 데이터이므로 interpolate나 mean 대신 ffill/bfill만 사용한다
    label = pd.DataFrame(label_rows, columns=["time", "label"]).set_index("time").sort_index()
    return (
        label.resample(RESAMPLE_RATE)
        .ffill(limit=INTERPOLATE_LIMIT)
        .bfill(limit=INTERPOLATE_LIMIT)
    )


def contiguous_window_starts(index: pd.DatetimeIndex, window: int, stride: int) -> np.ndarray:
    """결측 구간을 건너뛰지 않고 window 스텝이 연속으로 이어지는 윈도우의 시작 위치."""
    if len(index) < window:
        return np.array([], dtype=np.int64)
    step = pd.Timedelta(RESAMPLE_RATE).value
    t = index.asi8
    ok = (t[window - 1 :] - t[: len(t) - window + 1]) == (window - 1) * step
    starts = np.flatnonzero(ok)
    return starts[starts % stride == 0]


class TrainDataset(Dataset):
    def __init__(self, stride: int = 10):
        """
        :param stride: 윈도우 시작 간격(스텝). 기본 10스텝(100ms)마다 하나의 샘플을 만든다.
        """
        self.db = database.Database()

        signal = build_audio_frame(self.db.get_table_data("audio")).join(
            build_csi_frame(self.db.get_table_data("tcpdump"))
        )
        signal = signal.interpolate(limit=INTERPOLATE_LIMIT).dropna()
        data = signal.join(build_label_frame(self.db.get_table_data("label"))).dropna()

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
