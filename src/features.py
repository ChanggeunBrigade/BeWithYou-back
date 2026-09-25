"""
학습(dataload.py)과 실시간 추론(alerter.py)이 공유하는 입력 특징 정의.

두 경로가 반드시 같은 격자 간격, 같은 채널 순서로 텐서를 만들도록 여기서만 정의한다.
"""

N_SUBCARRIERS = 64
RESAMPLE_RATE = "10ms"  # 모델 입력 한 스텝의 간격
WINDOW_SIZE = 180  # 10ms * 180 = 1.8초
INTERPOLATE_LIMIT = 500  # 결측 보간 최대 스텝 수

AMPLITUDE_COLUMNS = [f"amplitudes_{sub}" for sub in range(N_SUBCARRIERS)]
PHASE_COLUMNS = [f"phases_{sub}" for sub in range(N_SUBCARRIERS)]
CSI_COLUMNS = AMPLITUDE_COLUMNS + PHASE_COLUMNS

# 모델 입력 채널 순서: 오디오 1채널 + 진폭 64채널 + 위상 64채널 = 129채널
FEATURE_COLUMNS = ["audio"] + CSI_COLUMNS
N_FEATURES = len(FEATURE_COLUMNS)


def csi_record_to_row(record: dict) -> dict | None:
    """esp.py가 보낸 CSI 레코드를 컬럼별 값으로 펼친다. 서브캐리어 수가 맞지 않으면 None."""
    amplitudes = record.get("amplitudes")
    phases = record.get("phases")
    if (
        amplitudes is None
        or phases is None
        or len(amplitudes) != N_SUBCARRIERS
        or len(phases) != N_SUBCARRIERS
    ):
        return None
    row = dict(zip(AMPLITUDE_COLUMNS, amplitudes))
    row.update(zip(PHASE_COLUMNS, phases))
    return row
