import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# `uv run start.py`로 실행하면 sys.executable은 프로젝트 가상환경(.venv)의 파이썬이다.
audio = subprocess.Popen(
    [sys.executable, "src/audio.py"],
    cwd=ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
esp = subprocess.Popen(
    [sys.executable, "src/esp.py"],
    cwd=ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    audio.terminate()
    esp.terminate()
