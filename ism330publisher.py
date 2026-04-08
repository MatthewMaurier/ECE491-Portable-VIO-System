#!/usr/bin/env python3
import time

import rospy
from sensor_msgs.msg import Imu

import board
import busio
from adafruit_lsm6ds.ism330dhcx import ISM330DHCX


def main():
    rospy.init_node("ism330dhcx_node", anonymous=False)

    # Parameters (settable via rosparam)
    i2c_address = rospy.get_param("~i2c_address", 0x6A)
    topic = rospy.get_param("~topic", "imu0")
    frame_id = rospy.get_param("~frame_id", "imu_link")
    rate_hz = float(rospy.get_param("~rate_hz", 100.0))

    # --- I2C + sensor (Pi 4 reliable init) ---
    i2c = busio.I2C(board.SCL, board.SDA)

    # Optional scan
    while not i2c.try_lock():
        pass
    found = [hex(x) for x in i2c.scan()]
    i2c.unlock()
    rospy.loginfo(f"Found I2C addresses: {found}")

    sensor = ISM330DHCX(i2c, address=int(i2c_address))
    rospy.loginfo(f"ISM330DHCX initialized at address 0x{int(i2c_address):02X}")

    pub = rospy.Publisher(topic, Imu, queue_size=200)
    rospy.loginfo(f"Publishing IMU on /{topic} at {rate_hz:.1f} Hz")

    r = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        try:
            ax, ay, az = sensor.acceleration   # m/s^2 (includes gravity)
            gx, gy, gz = sensor.gyro           # rad/s
        except Exception as e:
            rospy.logwarn(f"I2C read failed: {e}")
            r.sleep()
            continue

        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id

        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)

        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)

        # orientation left as default (unknown)
        pub.publish(msg)
        r.sleep()


if __name__ == "__main__":
    main()
