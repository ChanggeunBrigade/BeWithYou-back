import subprocess
import time

audio = subprocess.Popen(
    """cd /home/pi/BeWithYou-back/ && venv/bin/python src/audio.py""",
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
esp = subprocess.Popen(
    """cd /home/pi/BeWithYou-back/ && venv/bin/python src/esp.py""",
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    audio.terminate()
    esp.terminate()
