sudo tmux new -s docker docker compose up
sudo tmux new -s audio -d python3 audio/audio.py
sudo tmux new -s tcpdump -d python3 tcpdump/tcpdump.py
