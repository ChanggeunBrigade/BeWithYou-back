import base64
import datetime

import cv2
import numpy as np

import database

data = database.Database().get_table_data("opencv")

data.sort(key=lambda x: x[1])

i = 0
while i < len(data):
    row = data[i]
    frame = base64.b64decode(row[2]["frame"])
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, 1)
    frame = cv2.putText(
        frame,
        datetime.datetime.fromtimestamp(row[2]["ts"]).isoformat(),
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
        database.Database().insert_label(row[1], 0)  # 평상시
    if key == ord("s"):
        database.Database().insert_label(row[1], 1)  # 낙상
    if key == ord("z"):
        i -= 2
    i += 1
