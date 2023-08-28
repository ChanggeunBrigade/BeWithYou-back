import asyncio

import nats
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import database


class TrainDataset(Dataset):
    def __init__(self):
        self.db = database.Database()
        audio_rows = self.db.get_table_data("audio")
        csi_rows = self.db.get_table_data("tcpdump")
        resample_rate = "100ms"

        for i, row in enumerate(csi_rows):
            row2 = row[2]
            for sub in range(64):
                row2[f"amplitudes_{sub}"] = row2["amplitudes"][sub]
            for sub in range(64):
                row2[f"phases_{sub}"] = row2["phases"][sub]
            del row2["amplitudes"]
            del row2["phases"]

        audio_data = pd.DataFrame([i[2] for i in audio_rows])
        audio_data["ts"] = pd.to_datetime(audio_data["ts"], unit="s") + pd.Timedelta(
            hours=9
        )
        audio_data.set_index("ts", inplace=True)
        audio_data = audio_data.resample(resample_rate).mean().interpolate(limit=500)

        csi_data = pd.DataFrame([i[2] for i in csi_rows])
        csi_data["ts"] = pd.to_datetime(csi_data["ts"], unit="s") + pd.Timedelta(
            hours=9
        )
        csi_data.set_index("ts", inplace=True)
        csi_data = csi_data.resample(resample_rate).mean().interpolate(limit=500)

        signal_data: pd.DataFrame = (
            audio_data.join(csi_data).interpolate(limit=500).dropna()
        )

        # how to calculate y data? camera based labeling...
        # 카메라 라벨링을 수행할때는 interpolate나 mean은 쓰지 말고 ffill과 bfill만 사용해야 함. 0, 1 분류 데이터이기 때문
        label_rows = self.db.get_table_data("label")
        label: pd.DataFrame = pd.DataFrame(label_rows, columns=["time", "label"])

        label.set_index("time", inplace=True)
        label = (
            label.resample(resample_rate)
            .fillna(method="ffill", limit=500)
            .fillna(method="bfill", limit=500)
        )

        data = signal_data.join(label).dropna()
        self.x_data = list(
            map(
                lambda x: torch.from_numpy(x.drop("label", axis=1).to_numpy().T).type(
                    torch.FloatTensor
                ),
                filter(
                    lambda y: y.shape[0] >= 180,
                    list(data.rolling(window="1800ms", closed="right")),
                ),
            )
        )
        self.y_data = list(
            map(
                lambda x: (x["label"].to_numpy().sum() > 30).astype(np.float32),
                filter(
                    lambda y: y.shape[0] >= 180,
                    list(data.rolling(window="1800ms", closed="right")),
                ),
            )
        )

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]


if __name__ == "__main__":
    dataset = TrainDataset()
