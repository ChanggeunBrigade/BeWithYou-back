from math import isclose, pi

from esp import parse_csi


def test_parse_csi_line():
    data = parse_csi("CSI_DATA,1,aa:bb,-40,[3 4 0 -2 ]\n")
    assert data["amplitudes"] == [5.0, 2.0]
    assert isclose(data["phases"][1], pi)


def test_parse_csi_ignores_other_lines():
    assert parse_csi("I (1234) wifi: connected\n") is None
    assert parse_csi("[1 x 3]") is None
