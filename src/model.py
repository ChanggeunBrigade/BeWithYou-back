import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 데이터셋 로딩 모듈
def load_dataset():
    xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
    x_data = xy[:, 0:-1]
    y_data = xy[:, [-1]]
    return x_data, y_data
