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
    def split_dataset(dataset, val_ratio: float = 0.15, test_ratio: float = 0.15):
        """
        시간 순서로 앞에서부터 학습 / 검증 / 테스트 구간으로 나눈다 (기본 70 / 15 / 15).

        - 윈도우가 서로 크게 겹치므로 무작위로 나누면 거의 같은 윈도우가 여러 구간에 동시에
          들어가 성능이 부풀려진다. 구간 경계에 걸친 윈도우는 어느 쪽에도 넣지 않는다.
        - 검증 구간은 early stopping(모델 선택)에만 쓰고, 최종 성능은 한 번도 모델 선택에
          쓰지 않은 테스트 구간으로 보고한다.

        :return: (train, val, test) Subset과 학습 구간 끝 위치
        """
        length = dataset.x.shape[1]
        bounds = [
            0,
            int(length * (1 - val_ratio - test_ratio)),
            int(length * (1 - test_ratio)),
            length,
        ]
        starts = dataset.starts
        subsets = [
            Subset(dataset, np.flatnonzero((starts >= lo) & (starts + WINDOW_SIZE <= hi)))
            for lo, hi in zip(bounds, bounds[1:])
        ]
        return *subsets, bounds[1]

    def train_model(
        self,
        start=None,
        end=None,
        max_epochs: int = 1000,
        patience: int = 20,
        batch_size: int = 32,
    ):
        dataset = dataload.TrainDataset(start, end)
        train_data, val_data, test_data, train_end = self.split_dataset(dataset)
        if min(len(train_data), len(val_data), len(test_data)) == 0:
            raise RuntimeError(
                f"데이터가 부족합니다 (학습 {len(train_data)}, 검증 {len(val_data)}, "
                f"테스트 {len(test_data)})"
            )

        self.set_normalization(dataset.x[:, :train_end])
        self.to(device)
        model = nn.DataParallel(self) if torch.cuda.device_count() > 1 else self

        # 낙상 윈도우가 훨씬 적으므로 양성 샘플의 손실에 가중치를 준다
        train_y = dataset.y[train_data.indices]
        positives = train_y.sum()
        pos_weight = (len(train_y) - positives) / max(positives, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        optimizer = optim.NAdam(model.parameters(), lr=0.001)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

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

            val_loss, metrics = self.evaluate(val_loader, criterion)
            print(f"\nValidation: Average Loss: {val_loss:.4f}, {format_metrics(metrics)}\n")

            if val_loss < min_loss:
                min_loss = val_loss
                bad_epochs = 0
                torch.save(self.state_dict(), MODEL_PATH)
                print(f"saved : {min_loss}")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        print(f"best validation loss : {min_loss}")
        self.load_model()
        _, metrics = self.evaluate(DataLoader(test_data, batch_size=batch_size))
        print(f"Test: {format_metrics(metrics)}")
        return metrics

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

    def eval_model(self, start=None, end=None):
        """학습 때와 같은 기간을 주면 모델 선택에 쓰지 않은 테스트 구간으로 평가한다."""
        _, _, test_data, _ = self.split_dataset(dataload.TrainDataset(start, end))
        _, metrics = self.evaluate(DataLoader(test_data, batch_size=32))
        print(f"Test: {format_metrics(metrics)}")
        return metrics

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
    parser.add_argument("--start", help='데이터 기간 시작 (DB_TIMEZONE 기준, 예: "2024-05-01")')
    parser.add_argument("--end", help="데이터 기간 끝 (포함하지 않음)")
    args = parser.parse_args()

    model = Net()
    if args.command == "train":
        model.train_model(args.start, args.end)
    else:
        model.load_model()
        model.eval_model(args.start, args.end)
