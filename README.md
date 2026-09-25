# BeWithYou-back

Raspberry Pi 4B에서 작동하는 백엔드입니다.

백엔드 시스템의 구성은 다음과 같습니다.
1. 데이터 수집(audio.py, esp.py, opencv.py, labeler.py)
2. 데이터 처리(database.py, dataload.py)
3. 딥러닝 모델(model.py)
4. 신호 발송(alerter.py)

## 데이터 흐름

```
esp.py (CSI) ─┐                         ┌─ Kafka audio.data / tcpdump.data ─→ alerter.py ─→ MQTT alert
audio.py ─────┼─→ Fluent Bit(:30000) ───┤
opencv.py ────┘                         └─ PostgreSQL audio / tcpdump / opencv
                                              └─ labeler.py → label ─→ dataload.py → model.py(학습)
```

- Fluent Bit이 모든 레코드에 epoch 초 단위 `ts`를 붙인다.
- 학습(`dataload.py`)과 추론(`alerter.py`)은 `src/features.py`의 같은 함수로 10ms 격자,
  129채널(오디오 RMS 1 + CSI 진폭 64 + 위상 64) 입력을 만든다.
- PostgreSQL의 `time` 컬럼은 `DB_TIMEZONE`(기본 `Asia/Seoul`) 기준으로 저장된다.

## 모델 학습

```bash
uv run src/labeler.py        # opencv 프레임에 라벨 부여 (a: 평상시, s: 낙상, z: 이전, q: 종료)
uv run src/model.py train    # 저장소 루트에 model.pt 저장 (GPU가 있으면 사용)
uv run src/model.py eval     # 평가 구간 accuracy / precision / recall
```

- 입력: (129, 180) = 10ms × 1.8초 윈도우, 채널별 정규화 통계는 `model.pt`에 함께 저장
- 학습/평가 분할은 시간 순서(앞 80% / 뒤 20%)로 나눠 겹치는 윈도우가 섞이지 않게 함

## 인프라

```bash
cp .env.example .env   # 값 수정
docker compose up -d   # fluent-bit, kafka(KRaft), postgres, mosquitto
```

## 개발 환경

의존성은 [uv](https://docs.astral.sh/uv/)로 관리하며 `uv.lock`에 버전이 고정되어 있습니다.

```bash
# 공통 의존성 (학습/추론/라벨링)
uv sync

# 라즈베리파이에서 센서 수집까지 할 때 (gpiozero, pyserial, sounddevice)
sudo apt install libportaudio2   # sounddevice 런타임 의존성
uv sync --extra device

# 실행
uv run start.py
uv run src/alerter.py
```

테스트와 린트:

```bash
uv run pytest
uv run ruff check src tests
```

의존성을 추가/변경할 때는 `uv add <패키지>` 후 `uv.lock`을 함께 커밋합니다.
