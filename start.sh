docker compose up -d
sleep 10
tmux new -s audio -d /home/pi/venv/bin/python audio/audio.py
tmux new -s tcpdump -d /home/pi/venv/bin/python tcpdump/tcpdump.py
