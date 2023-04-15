import sys

import cv2
from fluent import asyncsender

FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트

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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    data = cv2.imencode(".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 100])[1].tolist()
    sender.emit("data", {"frame": data})
