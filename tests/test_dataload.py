import datetime
import math

import numpy as np
import pytest

import dataload
import features
from features import N_FEATURES, N_SUBCARRIERS, WINDOW_SIZE

START = 1_700_000_000.0


class FakeDatabase:
    """database.Database와 같은 결과를 내는 메모리 구현 (기간 필터는 무시)"""

    def __init__(self, tables):
        self.tables = tables

    def get_audio_power(self, step, start=None, end=None):
        buckets = {}
        for _, _, record in self.tables["audio"]:
            bucket = math.floor(record["ts"] / step)
            buckets.setdefault(bucket, []).append(features.audio_record_to_power(record))
        return [(bucket, float(np.mean(values))) for bucket, values in sorted(buckets.items())]

    def iter_csi(self, start=None, end=None):
        for _, _, record in self.tables["tcpdump"]:
            yield record["ts"], record["amplitudes"], record["phases"]

    def get_labels(self, start=None, end=None):
        return self.tables["label"]


def make_tables(seconds=5.0, fall_from=None):
    audio = [
        ("audio.data", None, {"ts": START + i / 1000, "data": float(np.sin(i))})
        for i in range(int(seconds * 1000))
    ]
    csi = [
        (
            "tcpdump.data",
            None,
            {
                "ts": START + i / 50,
                "amplitudes": [float(i)] * N_SUBCARRIERS,
                "phases": [0.5] * N_SUBCARRIERS,
            },
        )
        for i in range(int(seconds * 50))
    ]
    # 라벨 시각은 기존과 같이 +9시간 보정된 벽시계 시각으로 저장되어 있다고 가정
    base = datetime.datetime.fromtimestamp(START, datetime.UTC).replace(tzinfo=None)
    base += datetime.timedelta(hours=9)
    label = [
        (
            base + datetime.timedelta(milliseconds=100 * i),
            int(fall_from is not None and i / 10 >= fall_from),
        )
        for i in range(int(seconds * 10))
    ]
    return {"audio": audio, "tcpdump": csi, "label": label}


@pytest.fixture
def make_dataset(monkeypatch):
    def factory(**kwargs):
        tables = make_tables(**kwargs)
        monkeypatch.setattr(dataload.database, "Database", lambda: FakeDatabase(tables))
        return dataload.TrainDataset()

    return factory


def test_dataset_is_not_empty_and_has_model_shape(make_dataset):
    dataset = make_dataset(seconds=5.0)
    assert len(dataset) > 0
    x, y = dataset[0]
    assert tuple(x.shape) == (N_FEATURES, WINDOW_SIZE)
    assert y == 0.0


def test_channel_order_is_audio_then_amplitudes_then_phases(make_dataset):
    dataset = make_dataset(seconds=5.0)
    x, _ = dataset[0]
    # 위상 채널은 상수 0.5, 진폭은 증가, 오디오는 -1~1 범위
    assert np.allclose(x[1 + N_SUBCARRIERS :].numpy(), 0.5)
    assert x[1, -1] > x[1, 0]
    assert np.abs(x[0].numpy()).max() <= 1.0


def test_fall_label(make_dataset):
    dataset = make_dataset(seconds=5.0, fall_from=3.0)
    assert dataset.y[0] == 0.0
    assert dataset.y[-1] == 1.0


def test_windows_do_not_span_gaps():
    import pandas as pd

    index = pd.date_range("2024-01-01", periods=400, freq="10ms")
    index = index.delete(range(200, 210))  # 100ms 결측
    starts = features.contiguous_window_starts(index, WINDOW_SIZE, stride=1)
    for s in starts:
        assert index[s + WINDOW_SIZE - 1] - index[s] == pd.Timedelta("1790ms")


def test_timezone_aware_labels(monkeypatch):
    tables = make_tables(seconds=5.0, fall_from=3.0)
    kst = datetime.timezone(datetime.timedelta(hours=9))
    tables["label"] = [(t.replace(tzinfo=kst), v) for t, v in tables["label"]]
    monkeypatch.setattr(dataload.database, "Database", lambda: FakeDatabase(tables))
    dataset = dataload.TrainDataset()
    assert dataset.y[0] == 0.0 and dataset.y[-1] == 1.0


def test_legacy_raw_audio_becomes_rms(make_dataset):
    dataset = make_dataset(seconds=5.0)
    # 예전 형식(샘플 하나씩 `data`)도 10ms 구간 RMS로 변환된다: sin 파형이라 0보다 크다
    assert (dataset.x[0] > 0.1).all()
