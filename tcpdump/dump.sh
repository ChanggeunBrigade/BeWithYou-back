#!/bin/bash

interface="wlan0"

while true
do
        tcpdump -c 100 -i $interface -w - | python3 process.py | nc -N localhost 24224
done