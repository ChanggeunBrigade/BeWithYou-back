# BeWithYou-back

Raspberry Pi 4B에서 작동하는 백엔드입니다.

백엔드 시스템의 구성은 다음과 같습니다.
1. 데이터 수집(audio.py, esp.py, opencv.py, labeler.py)
2. 데이터 처리(database.py, dataload.py)
3. 딥러닝 모델(model.py)
4. 신호 발송(alerter.py)

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

의존성을 추가/변경할 때는 `uv add <패키지>` 후 `uv.lock`을 함께 커밋합니다.
