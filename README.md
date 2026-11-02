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
  라벨은 프레임의 epoch `ts`로 신호와 시각을 맞추므로 DB 시간대 설정과 무관하다.
- 센서 결측은 200ms까지만 보간한다. 그보다 긴 결측에 걸친 1.8초 윈도우는 학습/추론에 쓰지 않는다.

## 라즈베리파이 배포

`deploy/`의 systemd 유닛으로 수집기(`start.py`)와 알리미(`alerter.py`)를 띄운다.
두 프로세스 모두 종료되면 자동으로 다시 시작된다 (`start.py`도 자식 프로세스를 재시작).
라벨링용 카메라 수집(`opencv.py`)은 `.env`에 `COLLECT_VIDEO=true`를 준 동안에만 함께 뜬다.

```bash
# 부팅 시 tailscaled보다 Docker가 먼저 떠도 TAILSCALE_IP 포트 바인딩이 실패하지 않게 한다
sudo cp deploy/99-bewithyou-nonlocal-bind.conf /etc/sysctl.d/
sudo sysctl --system

sudo cp deploy/bewithyou-*.service deploy/bewithyou-*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now bewithyou-collector bewithyou-alerter bewithyou-retention.timer
```

알림 브로커 주소/계정/TLS, 알림 최소 간격(`ALERT_COOLDOWN`), 알림 판정 기준
(`ALERT_WINDOW`/`ALERT_VOTES`: 최근 10번 추론 중 5번 이상 낙상)은 `.env`로 설정한다.
TLS나 계정 없이 외부 브로커로 보내면 `alerter.py`가 시작할 때 경고를 남긴다.

`bewithyou-retention.timer`가 매일 오래된 데이터를 지운다 (`src/retention.py`).
카메라 프레임은 14일(`VIDEO_RETENTION_DAYS`), 오디오/CSI는 30일(`SIGNAL_RETENTION_DAYS`)이
지나면 지우되, 라벨 앞뒤 10초(`LABEL_MARGIN_SECONDS`) 데이터는 학습용으로 남긴다.

`esp.py`는 CSI가 계속 나오도록 기본 게이트웨이(공유기)로 10ms 간격 ping을 보내고
(`CSI_PING_TARGET`, `CSI_PING_INTERVAL`로 변경), CSI 수신율을 10초마다 로그로 남긴다.
초당 50건 미만이면 경고한다.

`alerter.py`는 추론하지 않는 경우와 그 이유를 10초마다 한 줄로 모아 로그로 남긴다.
- 센서 데이터가 3초 넘게 들어오지 않으면 추론하지 않는다 (지난 데이터로 알림을 보내지 않음).
- Kafka 오프셋을 저장하지 않으므로 재시작하면 항상 최신 데이터부터 읽는다.
- 서브캐리어 수가 64가 아닌 CSI(ESP32 CSI 설정 불일치)는 버리고 그 건수를 로그로 남긴다.
- 추론 한 번이 100ms를 넘기면 그 건수를 로그로 남긴다. 라즈베리파이에서
  `uv run src/model.py bench`로 추론 시간을 미리 확인할 수 있다.

## 모델 학습

```bash
# opencv 프레임에 라벨 부여
#   a/s: 현재 프레임 평상시/낙상, A/S: 현재부터 1초(10프레임) 평상시/낙상
#   d: 1초 건너뛰기, z: 이전, q: 종료. 라벨은 앞뒤 5초까지 이어 붙으므로 듬성듬성 달아도 된다
uv run src/labeler.py --start "2024-05-01 09:00" --end "2024-05-01 12:00"
# 저장소 루트에 model.pt 저장 (GPU가 있으면 사용). 기간을 생략하면 전체 데이터
# (데이터가 없는 공백 구간은 메모리에 펼치지 않으므로 메모리는 실제 데이터 양에 비례)
uv run src/model.py train --start 2024-05-01 --end 2024-06-01
# 학습 때와 같은 기간을 주면 테스트 구간의 accuracy / precision / recall 출력
uv run src/model.py eval --start 2024-05-01 --end 2024-06-01
```

- 입력: (129, 180) = 10ms × 1.8초 윈도우, 채널별 정규화 통계는 `model.pt`에 함께 저장
- 데이터를 시간 순서로 학습 70% / 검증 15% / 테스트 15%로 나눈다. 구간 경계에 걸친 윈도우는 버려
  겹치는 윈도우가 섞이지 않게 한다. 검증 구간은 early stopping에만 쓰고, 최종 성능은 테스트 구간으로 보고한다.
- 낙상 판정 임계값은 0.5로 고정하지 않고 검증 구간에서 F1이 가장 높은 값을 골라 `model.pt`에 저장한다
  (양성 가중치로 학습해 양성 쪽으로 기운 확률을 보정). 예전 `model.pt`는 0.5로 읽는다.

## 인프라

```bash
cp .env.example .env   # 값 수정 (TAILSCALE_IP는 `tailscale ip -4` 결과)
docker compose up -d   # fluent-bit, kafka(KRaft), postgres, mosquitto
```

### 네트워크 접근 제한 (Tailscale)

| 서비스 | 라즈베리파이 안 | 다른 PC (Tailscale) |
|---|---|---|
| Fluent Bit | `localhost:30000` | 열지 않음 |
| PostgreSQL | `localhost:5432` | `<TAILSCALE_IP>:5432` |
| Kafka | `localhost:9092` | `<TAILSCALE_IP>:9092` |
| Mosquitto | `localhost:1883` | `<TAILSCALE_IP>:1883` |

- 포트는 `127.0.0.1`과 `TAILSCALE_IP`에만 바인딩되므로 LAN 주소로는 접속할 수 없다.
- `deploy/tailscale-firewall.sh`는 여기에 더해 `TAILSCALE_IP`로 들어오는 새 연결 중
  출발지가 Tailscale 대역(`100.64.0.0/10`)이 아닌 것을 `DOCKER-USER` 체인에서 막는다.
  Docker가 공개한 포트는 ufw 같은 호스트 방화벽(INPUT 체인)을 거치지 않기 때문이다.
- 재부팅 직후 Tailscale 주소가 생기기 전에 Docker가 먼저 뜨면 바인딩에 실패하므로,
  아직 없는 주소에도 바인딩할 수 있게 설정해 둔다.

```bash
echo 'net.ipv4.ip_nonlocal_bind = 1' | sudo tee /etc/sysctl.d/99-bewithyou.conf
sudo sysctl --system
sudo cp deploy/tailscale-firewall.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now tailscale-firewall
```

학습용 PC의 `.env`에는 `PSQL_HOST=<TAILSCALE_IP 또는 MagicDNS 이름>`,
`KAFKA_BOOTSTRAP=<TAILSCALE_IP>:9092`를 넣는다.

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
