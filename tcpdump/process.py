import sys
import json
import base64
from datetime import datetime
from csi_reader import read_pcap

while True:
    # Read packet from stdin
    pcap = sys.stdin.read()
    if not pcap:
        break
    
    data = read_pcap(pcap)

    # Convert to base64 to ensure data integrity during transmission
    encoded_packet = base64.b64encode(pcap).decode()

    # Create a JSON record with a timestamp and the encoded packet data
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": encoded_packet
    }

    # Print the JSON record
    print(json.dumps(record))