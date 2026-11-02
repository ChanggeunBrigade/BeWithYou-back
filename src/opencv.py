import base64
import sys
import time

import cv2
from fluent import asyncsender

FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트
FRAME_INTERVAL = 0.1  # 라벨링용 프레임 전송 간격 (초), 라벨은 100ms 단위면 충분
# 라벨링에는 사람이 넘어지는지만 보이면 되므로 작게 줄여 저장한다 (DB 용량 절약)
FRAME_WIDTH = 320
WEBP_QUALITY = 60

sender = asyncsender.FluentSender(
    "opencv",
    host=FLUENT_BIT_HOST,
    port=FLUENT_BIT_PORT,
    nanosecond_precision=True,
    queue_maxsize=1000,
    queue_circular=True,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!", file=sys.stderr)
    sys.exit(1)

last_sent = 0.0
while True:
    ret, frame = cap.read()

    if not ret:
        # 비정상 종료로 알려 start.py가 다시 띄우게 한다
        print("Camera read failed!", file=sys.stderr)
        sys.exit(1)

    now = time.time()
    if now - last_sent < FRAME_INTERVAL:
        continue
    last_sent = now

    height, width = frame.shape[:2]
    if width > FRAME_WIDTH:
        frame = cv2.resize(frame, (FRAME_WIDTH, round(height * FRAME_WIDTH / width)))
    data = base64.b64encode(
        cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, WEBP_QUALITY])[1]
    )
    sender.emit_with_time("data", now, {"frame": data.decode("ascii")})
