from math import isclose, pi

from esp import RateMonitor, parse_csi


def test_parse_csi_line():
    data = parse_csi("CSI_DATA,1,aa:bb,-40,[3 4 0 -2 ]\n")
    assert data["amplitudes"] == [5.0, 2.0]
    assert isclose(data["phases"][1], pi)


def test_parse_csi_ignores_other_lines():
    assert parse_csi("I (1234) wifi: connected\n") is None
    assert parse_csi("[1 x 3]") is None


def test_rate_monitor():
    rate = RateMonitor(interval=10, min_rate=50)
    rate.started = 0.0
    assert all(rate.tick(i / 100) is None for i in range(999))
    assert rate.tick(10.0) == 100.0
    assert rate.flush(15.0) is None
    assert rate.flush(20.0) == 0.0
