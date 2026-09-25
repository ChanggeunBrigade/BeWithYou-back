import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

# 저장소 루트의 .env를 읽고, 같은 이름의 환경 변수가 있으면 그 값을 우선한다
config = {**dotenv_values(ROOT / ".env"), **os.environ}

# PostgreSQL의 `time` 컬럼(fluent-bit pgsql 출력, label 테이블)이 저장된 시간대
DB_TIMEZONE = config.get("DB_TIMEZONE", "Asia/Seoul")
KAFKA_BOOTSTRAP = config.get("KAFKA_BOOTSTRAP", "localhost:9092")

# 낙상 알림을 받을 MQTT 브로커
MQTT_HOST = config.get("MQTT_HOST", "osm-oracle.kro.kr")
MQTT_PORT = int(config.get("MQTT_PORT", "7001"))
MQTT_USER = config.get("MQTT_USER") or None
MQTT_PASS = config.get("MQTT_PASS") or None
MQTT_TLS = config.get("MQTT_TLS", "false").lower() in ("1", "true", "yes")
MQTT_TOPIC = config.get("MQTT_TOPIC", "alert")
# 낙상 상태가 이어지는 동안 알림을 반복해서 보내지 않도록 하는 최소 간격(초)
ALERT_COOLDOWN = float(config.get("ALERT_COOLDOWN", "30"))


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
