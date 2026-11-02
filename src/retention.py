"""
PostgreSQL에 쌓이는 센서 데이터를 정리한다 (deploy/bewithyou-retention.timer로 매일 실행).

- opencv 프레임: 라벨링에만 쓰므로 VIDEO_RETENTION_DAYS(기본 14일)가 지나면 지운다.
  라벨은 label 테이블에 epoch ts와 함께 남으므로 프레임을 지워도 학습에는 지장이 없다.
- 오디오/CSI: SIGNAL_RETENTION_DAYS(기본 30일)가 지나면 지우되, 라벨 앞뒤
  LABEL_MARGIN_SECONDS(기본 10초) 이내의 데이터는 학습용으로 남긴다.

값을 0으로 주면 그 테이블은 지우지 않는다.
"""

import logging

import database
from settings import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VIDEO_RETENTION_DAYS = float(config.get("VIDEO_RETENTION_DAYS", "14"))
SIGNAL_RETENTION_DAYS = float(config.get("SIGNAL_RETENTION_DAYS", "30"))
LABEL_MARGIN_SECONDS = float(config.get("LABEL_MARGIN_SECONDS", "10"))


def main():
    db = database.Database()
    plans = [
        ("opencv", VIDEO_RETENTION_DAYS, None),
        ("audio", SIGNAL_RETENTION_DAYS, LABEL_MARGIN_SECONDS),
        ("tcpdump", SIGNAL_RETENTION_DAYS, LABEL_MARGIN_SECONDS),
    ]
    for table, days, margin in plans:
        if days <= 0:
            continue
        deleted = db.delete_old(table, days, margin)
        log.info("%s: %g일 지난 행 %d개 삭제", table, days, deleted)


if __name__ == "__main__":
    main()
