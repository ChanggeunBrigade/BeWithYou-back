import torch.nn as nn
import torch.optim
from torch.utils.data import random_split

import src.dataload


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Conv1d(1, 10, kernel_size=3)
        self.lstm = nn.LSTM(10, 20, 3, batch_first=True)
        self.fc = nn.Linear(20, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# 데이터셋 로딩 모듈
def load_dataset():
    dataset = src.dataload.DataSet()
    train_dataset, test_dataset = random_split(dataset, [80, 20])

    return train_dataset, test_dataset


if __name__ == "__main__":
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters())

    train_data, test_data = load_dataset()

    for epoch in range(1000000):
        for i, (data, target) in enumerate(train_data):
            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        accuracy = model.evaluate(test_data)
        print("Epoch {}: Accuracy {:.2f}%".format(epoch + 1, accuracy * 100))
