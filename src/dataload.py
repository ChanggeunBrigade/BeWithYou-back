import pandas as pd
from torch.utils.data import Dataset

from src import database


class DataSet(Dataset):
    def __init__(self):
        self.db = database.Database()
        audio_rows = self.db.get_table_data("audio")
        csi_rows = self.db.get_table_data("tcpdump")

        for i in range(len(csi_rows)):
            row = csi_rows[i][2]
            for sub in range(64):
                row[f"amplitudes_{sub}"] = row["amplitudes"][sub]
            for sub in range(64):
                row[f"phases_{sub}"] = row["phases"][sub]
            del row["amplitudes"]
            del row["phases"]

        audio_data = pd.DataFrame([i[2] for i in audio_rows])
        audio_data["ts"] = pd.to_datetime(audio_data["ts"], unit="s")
        audio_data.set_index("ts", inplace=True)
        audio_data = audio_data.resample("10ms").mean().interpolate(limit=500)

        csi_data = pd.DataFrame([i[2] for i in csi_rows])
        csi_data["ts"] = pd.to_datetime(csi_data["ts"], unit="s")
        csi_data.set_index("ts", inplace=True)
        csi_data = csi_data.resample("10ms").mean().interpolate(limit=500)

        self.signal_data: pd.DataFrame = audio_data.join(csi_data)

        # how to calculate y data? camera based labeling...
        # 카메라 라벨링을 수행할때는 interpolate나 mean은 쓰지 말고 ffill과 bfill만 사용해야 함. 0, 1 분류 데이터이기 때문
        label_rows = self.db.get_table_data("label")
        self.label: pd.DataFrame = pd.DataFrame(label_rows, columns=["time", "label"])
        self.label.set_index("time", inplace=True)
        self.label = self.label.resample("10ms").fillna(method="ffill", limit=500).fillna(method="bfill", limit=500)

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, item):
        return self.signal_data[item], self.label[item]


if __name__ == "__main__":
    dataset = DataSet()
