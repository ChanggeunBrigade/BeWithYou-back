import serial
import sys
import gpiozero
import time
import subprocess

ping = subprocess.Popen(
    ["ping", "inc.sungkyul.ac.kr"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
)

con = serial.Serial("/dev/ttyUSB0", 921600)
rst = gpiozero.LED(2)

rst.off()
time.sleep(1)
rst.on()
time.sleep(1)

try:
    con.write(b"8\r\n")
except:
    pass


while True:
    try:
        data = con.readline().decode()
    except KeyboardInterrupt:
        rst.off()
        ping.kill()
        break
    except:
        continue
    print(data)
