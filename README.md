# BeWithYou-back
Raspberry Pi 3B+, 4 상에서 작동하는 백엔드입니다.

## Getting Started
1. 아래 링크를 따라 라즈베리파이 설정을 진행합니다.
https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92#getting-started

2. 아래 명령어를 통해 커널 업데이트를 방지합니다.
본 프로젝트는 5.10.92 이후 커널에서의 작동을 보장하지 않습니다.
```
sudo apt-mark hold libraspberrypi-bin libraspberrypi-dev libraspberrypi-doc libraspberrypi0
sudo apt-mark hold raspberrypi-bootloader raspberrypi-kernel raspberrypi-kernel-headers
```

3. docker를 설치합니다.
```
sudo apt install -y uidmap git
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
dockerd-rootless-setuptool.sh install
echo "export DOCKER_HOST=unix:///run/user/1000/docker.sock" | sudo tee -a /etc/bash.bashrc
```

## 작동 방법
아래 링크를 참조합니다.
https://github.com/nexmonster/nexmon_csi/tree/pi-5.10.92#usage

작동 결과로 pcap 파일이 반환됩니다.
(작성중)
