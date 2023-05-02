import subprocess
import time

audio = subprocess.Popen(
    """cd /home/pi/BeWithYou-back/ && venv/bin/python audio/audio.py""",
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
esp = subprocess.Popen(
    """cd /home/pi/BeWithYou-back/ && venv/bin/python esp/esp.py | venv/bin/python esp/parse.py""",
    shell=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

try:
    while True:
        time.sleep(0)
except KeyboardInterrupt:
    audio.terminate()
    esp.terminate()
