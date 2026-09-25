import torch
from test_dataload import FakeDatabase, make_tables

import dataload
import model
from features import N_FEATURES, WINDOW_SIZE


def test_forward_shape():
    net = model.Net()
    out = net(torch.randn(4, N_FEATURES, WINDOW_SIZE))
    assert out.shape == (4,)
    assert net.predict(torch.randn(1, N_FEATURES, WINDOW_SIZE)).shape == (1,)


def test_time_split_has_no_overlap(monkeypatch):
    tables = make_tables(seconds=20.0, fall_from=10.0)
    monkeypatch.setattr(dataload.database, "Database", lambda: FakeDatabase(tables))
    dataset = dataload.TrainDataset()
    train, test, split = model.Net.split_dataset(dataset)
    assert len(train) > 0 and len(test) > 0
    assert (dataset.starts[train.indices] + WINDOW_SIZE <= split).all()
    assert (dataset.starts[test.indices] >= split).all()


def test_train_saves_loadable_model(monkeypatch, tmp_path):
    tables = make_tables(seconds=20.0, fall_from=10.0)
    monkeypatch.setattr(dataload.database, "Database", lambda: FakeDatabase(tables))
    monkeypatch.setattr(model, "MODEL_PATH", tmp_path / "model.pt")

    net = model.Net()
    net.train_model(max_epochs=1)
    assert (tmp_path / "model.pt").exists()

    loaded = model.Net()
    loaded.load_model(tmp_path / "model.pt")
    # 정규화 통계도 함께 저장/복원된다
    assert torch.allclose(loaded.mean, net.mean.cpu())
    assert not torch.allclose(loaded.std, torch.ones_like(loaded.std))
