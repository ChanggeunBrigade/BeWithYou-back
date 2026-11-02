import logging
import re
import signal
import subprocess
import sys
import time
from math import atan2, sqrt

from fluent import sender

from settings import config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUDRATE = 921600
RESET_PIN = 2  # ESP32 리셋 GPIO
# CSI가 계속 나오도록 트래픽을 만드는 대상. 생략하면 기본 게이트웨이(공유기)로 보내
# 인터넷이 끊겨도 트래픽이 이어지게 한다
PING_TARGET = config.get("CSI_PING_TARGET") or None
PING_INTERVAL = config.get("CSI_PING_INTERVAL") or "0.01"  # 초당 100개
RATE_LOG_INTERVAL = 10.0  # CSI 수신율을 로그로 남기는 간격(초)
MIN_CSI_RATE = 50.0  # 초당 이보다 적게 들어오면 경고 (격자 10ms, 보간 한도 200ms)

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


def default_gateway() -> str | None:
    """`ip route show default`의 첫 게이트웨이 주소"""
    try:
        output = subprocess.run(
            ["ip", "route", "show", "default"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"default via (\S+)", output)
    return match.group(1) if match else None


def start_ping() -> subprocess.Popen | None:
    target = PING_TARGET or default_gateway()
    if target is None:
        log.warning("ping 대상(기본 게이트웨이)을 찾지 못함. CSI_PING_TARGET을 설정하세요")
        return None
    log.info("CSI 트래픽용 ping: %s (%s초 간격)", target, PING_INTERVAL)
    return subprocess.Popen(
        ["ping", "-i", PING_INTERVAL, target], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )


class RateMonitor:
    """CSI가 초당 몇 건 들어오는지 interval마다 로그로 남긴다."""

    def __init__(self, interval: float = RATE_LOG_INTERVAL, min_rate: float = MIN_CSI_RATE):
        self.interval = interval
        self.min_rate = min_rate
        self.count = 0
        self.started = time.monotonic()

    def tick(self, now: float) -> float | None:
        self.count += 1
        return self.flush(now)

    def flush(self, now: float) -> float | None:
        elapsed = now - self.started
        if elapsed < self.interval:
            return None
        rate = self.count / elapsed
        if rate < self.min_rate:
            log.warning("CSI 수신율 %.1f건/초 (최소 %g건/초 필요)", rate, self.min_rate)
        else:
            log.info("CSI 수신율 %.1f건/초", rate)
        self.count, self.started = 0, now
        return rate


def main():
    import gpiozero
    import serial

    # start.py가 terminate()로 종료시킬 때도 아래 finally에서 정리되도록 한다
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    logger = sender.FluentSender("tcpdump", host="localhost", port=30000, nanosecond_precision=True)
    ping = start_ping()
    ping_started = time.monotonic()
    rate = RateMonitor()
    rst = gpiozero.LED(RESET_PIN)
    try:
        # timeout: CSI가 아예 안 들어와도 readline이 돌아와 수신율 경고를 남길 수 있게 한다
        con = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)

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

            # 네트워크가 잠시 끊겨 ping이 종료됐으면 다시 띄운다
            ping_dead = ping is None or ping.poll() is not None
            if ping_dead and time.monotonic() - ping_started > RATE_LOG_INTERVAL:
                ping, ping_started = start_ping(), time.monotonic()

            data = parse_csi(line)
            if data is not None:
                logger.emit_with_time("data", time.time(), data)
                rate.tick(time.monotonic())
            else:
                rate.flush(time.monotonic())
    except KeyboardInterrupt:
        pass
    finally:
        rst.off()
        if ping is not None:
            ping.kill()
        logger.close()


if __name__ == "__main__":
    main()
