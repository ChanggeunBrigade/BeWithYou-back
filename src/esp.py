import subprocess
import time
import sys
import re
from math import sqrt, atan2

import gpiozero
import serial

from fluent import sender

logger = sender.FluentSender(
    "tcpdump", host="localhost", port=30000, nanosecond_precision=True
)

ping = subprocess.Popen(
    ["ping", "inc.sungkyul.ac.kr"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)

con = serial.Serial("/dev/ttyUSB0", 921600)
rst = gpiozero.LED(2)

rst.off()
time.sleep(1)
rst.on()
time.sleep(1)

try:
    con.write(b"8\r\n")
except:
    pass

while True:
    try:
        data = con.readline().decode()
    except KeyboardInterrupt:
        rst.off()
        ping.kill()
        break
    except:
        continue

    imaginary = []
    real = []
    amplitudes = []
    phases = []

    # Parse string to create integer list
    csi_string = re.findall(r"\[(.*)\]", data)
    if csi_string:
        csi_string = csi_string[0]
    else:
        continue
    try:
        csi_raw = [int(x) for x in csi_string.split(" ") if x != ""]

        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw)):
            if i % 2 == 0:
                imaginary.append(csi_raw[i])
            else:
                real.append(csi_raw[i])

        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw) / 2)):
            amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
            phases.append(atan2(imaginary[i], real[i]))

        data = {"amplitudes": amplitudes, "phases": phases}

        # sys.stdout.write("{} {}\n".format(time.time(), data))

        logger.emit_with_time("data", time.time(), data)
    except:
        pass
