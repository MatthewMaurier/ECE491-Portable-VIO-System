import time
import board

from adafruit_lsm6ds.ism330dhcx import ISM330DHCX

# Create I2C bus using the default SCL/SDA pins (/dev/i2c-1 on Pi)
i2c = board.I2C()  # uses board.SCL and board.SDA

# Optional: debug scan to prove we see 0x6A
while not i2c.try_lock():
    pass
print("Found I2C addresses:", [hex(x) for x in i2c.scan()])
i2c.unlock()

# Create sensor object at 0x6A (what i2cdetect showed)
sensor = ISM330DHCX(i2c, address=0x6A)

print("ISM330DHCX initialized OK, reading values...\n")

while True:
    ax, ay, az = sensor.acceleration   # m/s^2
    gx, gy, gz = sensor.gyro           # rad/s

    print(
        f"Accel: X={ax:6.2f} Y={ay:6.2f} Z={az:6.2f} m/s^2  |  "
        f"Gyro: X={gx:6.2f} Y={gy:6.2f} Z={gz:6.2f} rad/s"
    )
    time.sleep(0.02)