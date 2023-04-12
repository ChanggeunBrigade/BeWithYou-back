import os
import sys
from fluent import sender
from csi_reader import read_pcap

FLUENT_BIT_HOST = "localhost"  # Fluent Bit 서버의 IP 주소 또는 호스트명
FLUENT_BIT_PORT = 30000  # Fluent Bit 서버의 포트
PCAP_COUNT = 100
PCAP_PATH = "tmp/csi.pcap"

logger = sender.FluentSender(
    "tcpdump", host=FLUENT_BIT_HOST, port=FLUENT_BIT_PORT, nanosecond_precision=True
)


while True:
    os.system("sudo rm -rf {}".format(PCAP_PATH))
    cmd = "sudo timeout 10 tcpdump -i wlan0 dst port 5500 -w {}".format(PCAP_PATH)

    os.system(cmd)

    with open(PCAP_PATH, "rb") as f:
        data = f.read()

    data = read_pcap(data)

    for i in range(data.nsamples):
        timestamp = float(data.get_timestamp(i))
        rssi = int(data.get_rssi(i))
        csi = data.get_csi(i)

        logger.emit_with_time(
            "data",
            timestamp,
            {"rssi": rssi, "csi": list(map(lambda x: str(x), csi.tolist()))},
        )
