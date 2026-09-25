import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import dataload
from features import N_FEATURES, WINDOW_SIZE
from settings import ROOT

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

MODEL_PATH = ROOT / "model.pt"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 학습 데이터에서 계산한 채널별 평균/표준편차. state_dict에 함께 저장되어 추론에도 쓰인다
        self.register_buffer("mean", torch.zeros(N_FEATURES, 1))
        self.register_buffer("std", torch.ones(N_FEATURES, 1))
        self.cnn = nn.Conv1d(N_FEATURES, 200, kernel_size=3)
        self.lstm = nn.LSTM(200, 300, 3, batch_first=True)
        self.fc = nn.Linear(300, 1)

    def forward(self, x):
        """
        딥러닝 연산을 수행합니다.

        :param x: (batch, 129, 180)의 10ms 단위 시계열 데이터
        :return: (batch,) 낙상 logit (sigmoid를 거치면 0~1 확률)
        """
        x = (x - self.mean) / self.std
        x = self.cnn(x)  # (batch, 200, 178)
        x, _ = self.lstm(x.transpose(1, 2))  # 시간축(178)을 따라 LSTM을 돌린다
        return self.fc(x[:, -1]).squeeze(-1)

    def set_normalization(self, x: torch.Tensor):
        """x: (채널, 시간) 학습 구간 전체 시계열"""
        self.mean.copy_(x.mean(dim=1, keepdim=True))
        self.std.copy_(x.std(dim=1, keepdim=True).clamp_min(1e-6))

    @staticmethod
    def split_dataset(dataset, test_ratio: float = 0.2):
        """
        시간 순서로 앞 80%를 학습, 뒤 20%를 평가에 쓴다.

        윈도우가 서로 크게 겹치므로 무작위로 나누면 거의 같은 윈도우가 학습/평가에 동시에
        들어가 정확도가 부풀려진다. 경계에 걸친 윈도우는 어느 쪽에도 넣지 않는다.
        """
        split = int(dataset.x.shape[1] * (1 - test_ratio))
        starts = dataset.starts
        train_idx = np.flatnonzero(starts + WINDOW_SIZE <= split)
        test_idx = np.flatnonzero(starts >= split)
        return Subset(dataset, train_idx), Subset(dataset, test_idx), split

    def train_model(self, max_epochs: int = 1000, patience: int = 20, batch_size: int = 32):
        dataset = dataload.TrainDataset()
        train_data, test_data, split = self.split_dataset(dataset)
        if len(train_data) == 0 or len(test_data) == 0:
            raise RuntimeError("학습/평가 데이터가 부족합니다")

        self.set_normalization(dataset.x[:, :split])
        self.to(device)
        model = nn.DataParallel(self) if torch.cuda.device_count() > 1 else self

        # 낙상 윈도우가 훨씬 적으므로 양성 샘플의 손실에 가중치를 준다
        train_y = dataset.y[train_data.indices]
        positives = train_y.sum()
        pos_weight = (len(train_y) - positives) / max(positives, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        optimizer = optim.NAdam(model.parameters(), lr=0.001)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        min_loss = float("inf")
        bad_epochs = 0

        for epoch in range(max_epochs):
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print(
                        f"Train Epoch: {epoch} [{i * len(x)}/{len(train_data)} "
                        f"({100 * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                    )

            test_loss, metrics = self.evaluate(test_loader, criterion)
            print(f"\nTest set: Average Loss: {test_loss:.4f}, {format_metrics(metrics)}\n")

            if test_loss < min_loss:
                min_loss = test_loss
                bad_epochs = 0
                torch.save(self.state_dict(), MODEL_PATH)
                print(f"saved : {min_loss}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        print(f"best loss : {min_loss}")

    @torch.no_grad()
    def evaluate(self, loader, criterion=None):
        self.eval()
        total_loss, tp, fp, fn, tn = 0.0, 0, 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = self(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * len(y)
            pred = (logits > 0).float()
            tp += int(((pred == 1) & (y == 1)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
            tn += int(((pred == 0) & (y == 0)).sum())
        total = max(tp + fp + fn + tn, 1)
        metrics = {
            "accuracy": (tp + tn) / total,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
        return total_loss / total, metrics

    def eval_model(self):
        _, test_data, _ = self.split_dataset(dataload.TrainDataset())
        _, metrics = self.evaluate(DataLoader(test_data, batch_size=32))
        print(format_metrics(metrics))

    def predict(self, data, threshold: float = 0.5):
        with torch.no_grad():
            return (torch.sigmoid(self(data.to(device))) > threshold).float()

    def load_model(self, path=None):
        self.load_state_dict(torch.load(path or MODEL_PATH, map_location=device))
        self.to(device)


def format_metrics(metrics: dict) -> str:
    return ", ".join(f"{name}: {value:.3f}" for name, value in metrics.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="낙상 감지 모델 학습/평가")
    parser.add_argument("command", choices=["train", "eval"])
    args = parser.parse_args()

    model = Net()
    if args.command == "train":
        model.train_model()
    else:
        model.load_model()
        model.eval_model()
