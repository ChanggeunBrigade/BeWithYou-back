import re
import sys
import time
from math import sqrt, atan2

from fluent import sender

logger = sender.FluentSender(
    "tcpdump", host="localhost", port=30000, nanosecond_precision=True
)

if __name__ == "__main__":
    while True:
        line = sys.stdin.readline()

        imaginary = []
        real = []
        amplitudes = []
        phases = []

        # Parse string to create integer list
        csi_string = re.findall(r"\[(.*)\]", line)
        if csi_string:
            csi_string = csi_string[0]
        else:
            continue
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

        # sys.stdout.write("{}\n".format(data))

        logger.emit_with_time("data", time.time(), data)
