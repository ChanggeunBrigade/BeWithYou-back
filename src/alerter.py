import asyncio
import json
import queue

import nats
import paho.mqtt.client as mqtt_client
import pandas

import model

data_queue = queue.Queue()


async def nats_msg_handler(msg):
    data = msg.data.decode()
    parsed = json.loads(data)
    data_queue.put(parsed)


if __name__ == "__main__":

    def main():
        classifier = model.Net()
        classifier.load_model()

        alerter = mqtt_client.Client("alerter", protocol=mqtt_client.MQTTv5)
        alerter.connect("mqtt://osm-oracle.kro.kr", port=7001)

        nats_conn = asyncio.run(nats.connect("nats://osm-oracle:7005"))
        asyncio.run(nats_conn.subscribe("data", cb=nats_msg_handler))

        data = pandas.DataFrame(
            columns=[[f"amplitudes_{sub}" for sub in range(64)] + [f"phases_{sub}" for sub in range(64)] + ["audio"]]
        )
        while True:
            while len(data) < 180:
                data.append(data_queue.get())
            predicted = classifier.predict(data)
            if predicted == 1:
                alerter.publish("alert", b"alert")
            data = data[10:]

    main()
