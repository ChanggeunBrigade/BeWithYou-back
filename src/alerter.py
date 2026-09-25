import json
import logging

import paho.mqtt.client as mqtt_client
import pandas as pd
import torch
from kafka import KafkaConsumer

import model
from features import FEATURE_COLUMNS, WINDOW_SIZE, csi_record_to_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    def main():
        classifier = model.Net()
        classifier.load_model()
        classifier.eval()

        consumer = KafkaConsumer(
            "data",
            group_id="alert",
            bootstrap_servers="deeprasp:9092",
            auto_offset_reset="latest",
        )

        alerter = mqtt_client.Client("alerter", protocol=mqtt_client.MQTTv5)
        alerter.connect("osm-oracle.kro.kr", port=7001)
        alerter.loop_start()

        data = pd.DataFrame(columns=FEATURE_COLUMNS, dtype="float32")

        for message in consumer:
            try:
                value = json.loads(message.value.decode("utf-8"))
                row = csi_record_to_row(value)
                if row is None:
                    logger.warning("CSI 서브캐리어 수가 맞지 않아 건너뜀")
                    continue
                row["audio"] = value.get("audio", 0)
                data.loc[len(data)] = [row[column] for column in FEATURE_COLUMNS]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("데이터 파싱 오류: %s", e)
                continue

            if len(data) >= WINDOW_SIZE:
                try:
                    tensor = torch.from_numpy(
                        data.tail(WINDOW_SIZE).to_numpy(dtype="float32").T.copy()
                    ).unsqueeze(0).float()
                    with torch.no_grad():
                        predicted = classifier.predict(tensor)
                    if predicted.item() == 1:
                        alerter.publish("alert", b"alert")
                        logger.info("낙상 감지 알림 전송")
                except Exception as e:
                    logger.error("추론 오류: %s", e)
                data = data.iloc[10:].reset_index(drop=True)

    main()
