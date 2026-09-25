#!/bin/sh
# Docker가 공개한 포트에 Tailscale 대역(100.64.0.0/10)에서 온 연결만 허용한다.
#
# docker-compose.yml은 PostgreSQL/Kafka/Mosquitto를 127.0.0.1과 TAILSCALE_IP에만 바인딩한다.
# 이 스크립트는 여기에 더해, LAN의 다른 호스트가 라즈베리파이를 게이트웨이 삼아
# TAILSCALE_IP로 직접 패킷을 보내는 경우까지 막는다. Docker 포트는 호스트의 INPUT 체인
# (ufw 등)을 거치지 않으므로 DOCKER-USER 체인에 규칙을 넣어야 한다.
#
# 사용법: sudo deploy/tailscale-firewall.sh [TAILSCALE_IP]  (생략 시 `tailscale ip -4`)
set -eu

TAILSCALE_CIDR="100.64.0.0/10"
TAILSCALE_IP="${1:-$(tailscale ip -4 | head -n 1)}"

# DOCKER-USER 체인은 dockerd가 만든다. 규칙이 이미 있으면 다시 넣지 않는다.
rule="-m conntrack --ctstate NEW --ctorigdst ${TAILSCALE_IP} ! -s ${TAILSCALE_CIDR} -j DROP"
# shellcheck disable=SC2086
if ! iptables -C DOCKER-USER $rule 2>/dev/null; then
    iptables -I DOCKER-USER $rule
fi
echo "DOCKER-USER: ${TAILSCALE_IP} 로의 새 연결은 ${TAILSCALE_CIDR} 에서만 허용"
