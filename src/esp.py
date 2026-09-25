import logging
import re
import signal
import subprocess
import sys
import time
from math import atan2, sqrt

from fluent import sender

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 921600
RESET_PIN = 2  # ESP32 리셋 GPIO
PING_TARGET = "inc.sungkyul.ac.kr"  # CSI가 계속 나오도록 트래픽을 만드는 대상

CSI_PATTERN = re.compile(r"\[(.*)\]")


def parse_csi(line: str) -> dict | None:
    """ESP32 CSI 한 줄([imag real imag real ...])을 진폭/위상으로 변환한다."""
    match = CSI_PATTERN.search(line)
    if not match:
        return None
    try:
        csi_raw = [int(x) for x in match.group(1).split(" ") if x != ""]
    except ValueError as e:
        log.warning("CSI 파싱 오류: %s", e)
        return None

    imaginary = csi_raw[0::2]
    real = csi_raw[1::2]
    pairs = list(zip(imaginary, real))
    return {
        "amplitudes": [sqrt(i**2 + r**2) for i, r in pairs],
        "phases": [atan2(i, r) for i, r in pairs],
    }


def main():
    import gpiozero
    import serial

    # start.py가 terminate()로 종료시킬 때도 아래 finally에서 정리되도록 한다
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    logger = sender.FluentSender("tcpdump", host="localhost", port=30000, nanosecond_precision=True)
    ping = subprocess.Popen(
        ["ping", PING_TARGET], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    rst = gpiozero.LED(RESET_PIN)
    try:
        con = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE)

        rst.off()
        time.sleep(1)
        rst.on()
        time.sleep(1)

        try:
            con.write(b"8\r\n")
        except serial.SerialException as e:
            log.warning("시리얼 초기화 실패: %s", e)

        while True:
            try:
                line = con.readline().decode()
            except UnicodeDecodeError as e:
                log.warning("시리얼 읽기 오류: %s", e)
                continue
            # SerialException(장치 분리 등)은 그대로 종료하고 start.py가 다시 띄운다

            data = parse_csi(line)
            if data is not None:
                logger.emit_with_time("data", time.time(), data)
    except KeyboardInterrupt:
        pass
    finally:
        rst.off()
        ping.kill()
        logger.close()


if __name__ == "__main__":
    main()
