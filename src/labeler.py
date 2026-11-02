import argparse
import base64
import datetime
from zoneinfo import ZoneInfo

import cv2
import numpy as np

import database
from settings import DB_TIMEZONE

FRAMES_PER_SECOND = 10  # opencv.py의 FRAME_INTERVAL(0.1초)에 맞춘 1초 분량 프레임 수
KEYS = """\
a: 평상시, s: 낙상 (현재 프레임 하나)
A: 평상시, S: 낙상 (현재부터 1초 = 10프레임)
d: 1초 건너뛰기, z: 이전, q: 종료, 그 외: 다음"""

parser = argparse.ArgumentParser(
    description="opencv 프레임 라벨링\n" + KEYS, formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("--start", help='기간 시작 (DB_TIMEZONE 기준, 예: "2024-05-01 09:00")')
parser.add_argument("--end", help="기간 끝 (포함하지 않음)")
args = parser.parse_args()

db = database.Database()
# 프레임 이미지는 전부 올리지 않고 시각 목록만 읽은 뒤 화면에 띄울 때 한 장씩 가져온다
times = db.get_frame_times(args.start, args.end)
print(f"frames: {len(times)}\n{KEYS}")
tz = ZoneInfo(DB_TIMEZONE)


def frame_ts(time) -> float | None:
    record = db.get_frame(time)
    return record.get("ts") if record else None


i = 0
while i < len(times):
    record = db.get_frame(times[i])
    if record is None or "frame" not in record:
        i += 1  # 그 사이 정리(retention.py)된 프레임
        continue
    frame = base64.b64decode(record["frame"])
    frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), 1)
    # 라벨링하는 PC의 시간대가 아니라 DB 시간대(--start/--end와 같은 기준)로 표시한다
    shown = datetime.datetime.fromtimestamp(record["ts"], tz).strftime("%Y-%m-%d %H:%M:%S.%f")
    frame = cv2.putText(frame, shown[:-4], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow("1", frame)
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
    if key in (ord("a"), ord("s")):
        db.insert_labels([(times[i], record["ts"])], int(key == ord("s")))
    elif key in (ord("A"), ord("S")):
        span = times[i : i + FRAMES_PER_SECOND]
        db.insert_labels([(t, frame_ts(t)) for t in span], int(key == ord("S")))
        i += len(span)
        continue
    elif key == ord("d"):
        i += FRAMES_PER_SECOND
        continue
    elif key == ord("z"):
        i = max(i - 1, 0)  # 이전 프레임으로
        continue
    i += 1
