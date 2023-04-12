docker compose up -d
sleep 10
tmux new -s audio -d python3 audio/audio.py
tmux new -s tcpdump -d python3 tcpdump/tcpdump.py
