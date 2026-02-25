import json
import logging

import paho.mqtt.client as mqtt_client
import pandas as pd
import torch
from kafka import KafkaConsumer

import model

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

        columns = (
            [f"amplitudes_{sub}" for sub in range(64)]
            + [f"phases_{sub}" for sub in range(64)]
            + ["audio"]
        )
        data = pd.DataFrame(columns=columns)

        for message in consumer:
            try:
                value = json.loads(message.value.decode("utf-8"))
                row = {}
                for sub in range(64):
                    row[f"amplitudes_{sub}"] = value.get("amplitudes", [0] * 64)[sub]
                    row[f"phases_{sub}"] = value.get("phases", [0] * 64)[sub]
                row["audio"] = value.get("audio", 0)
                data = pd.concat(
                    [data, pd.DataFrame([row])], ignore_index=True
                )
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.warning("데이터 파싱 오류: %s", e)
                continue

            if len(data) >= 180:
                try:
                    tensor = torch.from_numpy(
                        data.tail(180).to_numpy().T
                    ).unsqueeze(0).float()
                    with torch.no_grad():
                        predicted = classifier.predict(tensor)
                    if predicted.item() == 1:
                        alerter.publish("alert", b"alert")
                        logger.info("낙상 감지 알림 전송")
                except Exception as e:
                    logger.error("추론 오류: %s", e)
                data = data.iloc[10:]

    main()
