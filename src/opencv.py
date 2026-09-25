import base64
import sys
import time

import cv2
from fluent import asyncsender

FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트
FRAME_INTERVAL = 0.1  # 라벨링용 프레임 전송 간격 (초), 라벨은 100ms 단위면 충분

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
    print("Camera open failed!")
    sys.exit(0)

last_sent = 0.0
while True:
    ret, frame = cap.read()

    if not ret:
        break

    now = time.time()
    if now - last_sent < FRAME_INTERVAL:
        continue
    last_sent = now

    data = base64.b64encode(cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 100])[1])
    sender.emit_with_time("data", now, {"frame": data.decode("ascii")})
