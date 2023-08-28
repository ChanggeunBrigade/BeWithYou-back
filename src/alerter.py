import paho.mqtt.client as mqtt_client
import pandas as pd
from kafka import KafkaConsumer

import model


if __name__ == "__main__":

    def main():
        classifier = model.Net()
        classifier.load_model()

        consumer = KafkaConsumer(
            "data",
            group_id="alert",
            bootstrap_servers="deeprasp:9092",
            auto_offset_reset="latest",
        )

        alerter = mqtt_client.Client("alerter", protocol=mqtt_client.MQTTv5) 
        alerter.connect("mqtt://osm-oracle.kro.kr", port=7001)
        data = pd.DataFrame(
            columns=[
                [f"amplitudes_{sub}" for sub in range(64)]
                + [f"phases_{sub}" for sub in range(64)]
                + ["audio"]
            ]
        )
        while True:
            while len(data) < 180:
                predicted = classifier.predict(data)
                if predicted == 1:
                    alerter.publish("alert", b"alert")
                data = data[10:]

    main()
