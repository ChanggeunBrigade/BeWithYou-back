import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ["src/audio.py", "src/esp.py"]
RESTART_DELAY = 5  # 죽은 프로세스를 다시 띄우기 전 대기 시간(초)


def spawn(script: str) -> subprocess.Popen:
    # `uv run start.py`로 실행하면 sys.executable은 프로젝트 가상환경(.venv)의 파이썬이다.
    print(f"starting {script}", flush=True)
    return subprocess.Popen([sys.executable, script], cwd=ROOT)


def main():
    # systemd 등이 SIGTERM으로 종료시킬 때도 자식 프로세스를 정리한다
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    processes = {script: spawn(script) for script in SCRIPTS}
    died_at: dict[str, float] = {}
    try:
        while True:
            time.sleep(1)
            for script, process in processes.items():
                if process.poll() is None:
                    continue
                died_at.setdefault(script, time.monotonic())
                if time.monotonic() - died_at[script] >= RESTART_DELAY:
                    print(f"{script} exited with {process.returncode}, restarting", flush=True)
                    processes[script] = spawn(script)
                    del died_at[script]
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
