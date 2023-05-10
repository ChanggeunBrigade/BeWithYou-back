import pandas as pd
from torch.utils.data import Dataset

from src import database


class DataSet(Dataset):
    def __init__(self):
        self.db = database.Database()
        audio_rows = self.db.get_table_data("audio")
        video_data = self.db.get_table_data("opencv")
        csi_rows = self.db.get_table_data("tcpdump")

        for i in range(len(csi_rows)):
            row = csi_rows[i][2]
            for sub in range(len(row['amplitudes'])):
                row[f'amplitudes_{sub}'] = row['amplitudes'][sub]
            for sub in range(len(row['phases'])):
                row[f'phases_{sub}'] = row['phases'][sub]
            del row['amplitudes']
            del row['phases']

        audio_data = pd.DataFrame([i[2] for i in audio_rows])
        audio_data.set_index('ts', inplace=True)
        audio_data = audio_data.resample('100ms').mean().interpolate()

        csi_data = pd.DataFrame([i[2] for i in csi_rows])
        csi_data.set_index('ts', inplace=True)
        csi_data = csi_data.resample('100ms').mean().interpolate()

        self.signal_data = audio_data.join(csi_data).mean().interpolate().ffill().bfill()

        # how to calculate y data? camera based labeling...
        # 카메라 라벨링을 수행할때는 interpolate나 mean은 쓰지 말고 ffill과 bfill만 사용해야 함. 0, 1 분류 데이터이기 때문

    def __len__(self):
        return

    def __getitem__(self, idx: int):
        return
