import argparse
import base64
import datetime

import cv2
import numpy as np

import database

parser = argparse.ArgumentParser(
    description="opencv 프레임 라벨링 (a: 평상시, s: 낙상, z: 이전, 그 외: 다음, q: 종료)"
)
parser.add_argument("--start", help='기간 시작 (DB_TIMEZONE 기준, 예: "2024-05-01 09:00")')
parser.add_argument("--end", help="기간 끝 (포함하지 않음)")
args = parser.parse_args()

db = database.Database()
# 프레임 이미지는 전부 올리지 않고 시각 목록만 읽은 뒤 화면에 띄울 때 한 장씩 가져온다
times = db.get_frame_times(args.start, args.end)
print(f"frames: {len(times)}")

i = 0
while i < len(times):
    record = db.get_frame(times[i])
    frame = base64.b64decode(record["frame"])
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, 1)
    frame = cv2.putText(
        frame,
        datetime.datetime.fromtimestamp(record["ts"]).isoformat(),
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
    )
    cv2.imshow("1", frame)
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    if key == ord("a"):
        db.insert_label(times[i], 0)  # 평상시
    if key == ord("s"):
        db.insert_label(times[i], 1)  # 낙상
    if key == ord("z"):
        i = max(i - 1, 0)  # 이전 프레임으로
        continue
    i += 1
