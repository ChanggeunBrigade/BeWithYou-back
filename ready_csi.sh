#!/bin/bash

mkdir tmp
sudo mount -t tmpfs -o size=50M tmpfs tmp

TARGET_INTERFACE="wlan0"
MONITOR_INTERFACE="mon0"
MCP_CORE_MASK=1
MCP_STREAM_MASK=1
MCP_CHANNEL_BANDWIDTH="8/20"
MCP=$("mcp" "-C" $MCP_CORE_MASK "-N" $MCP_STREAM_MASK "-c" $MCP_CHANNEL_BANDWIDTH)

if ifconfig "$MONITOR_INTERFACE" > /dev/null 2>&1; then
    echo "이미 인터페이스가 존재함. 재부팅 후 재실행 필요."
    exit 1
fi

sudo ifconfig $TARGET_INTERFACE up
sudo nexutil -I$TARGET_INTERFACE -s500 -b -l34 -v$MCP
sudo iw dev $TARGET_INTERFACE interface add mon0 type monitor
sudo ip link set $MONITOR_INTERFACE up
