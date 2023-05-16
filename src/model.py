import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

import dataload

torch.manual_seed(42)

cuda0 = torch.device("cuda:0")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = nn.Conv1d(129, 200, kernel_size=3)
        self.lstm = nn.LSTM(178, 300, 3, batch_first=True)
        self.fc1 = nn.Linear(300, 1)
        self.fc2 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = x.squeeze()
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 데이터셋 로딩 모듈
def load_dataset():
    dataset = dataload.DataSet()
    data_size = len(dataset)
    train_size = int(data_size * 0.8)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def train():
    model = Net()
    model = nn.DataParallel(model)
    model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.001)

    train_data, test_data = load_dataset()

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    min_loss = 1e9
    early_stop_counter = 5000

    for epoch in range(1000000):
        model.module.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.float()
            x = x.to(cuda0)
            y = y.to(cuda0)
            y = y.unsqueeze(0).T

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        i * len(x),
                        len(train_loader.dataset),
                        100 * i / len(train_loader),
                        loss.item(),
                    )
                )

        model.module.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(cuda0), y.to(cuda0)
                y = y.unsqueeze(0).T
                output = model(x)
                test_loss += criterion(output, y).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.module.state_dict(), "./model.pt")
            print("saved : {}".format(min_loss))
        else:
            early_stop_counter -= 1
            if early_stop_counter == 0:
                print("best loss : {}".format(min_loss))
                break


def evaluate():
    model = Net()
    model.load_state_dict(torch.load("./model.pt"))
    _, test_data = load_dataset()
    test_loader = DataLoader(test_data, batch_size=32)
    model.eval()

    correct = 0
    total = 0
    for inputs, labels in test_loader:
        labels = labels.unsqueeze(0).T
        predictions = model(inputs)
        correct += (predictions.round() == labels.round()).sum().item()
        total += len(labels)

    print("Accuracy: {}".format(correct / total))


def predict(model, data):
    return model(data).round()


if __name__ == "__main__":
    evaluate()
