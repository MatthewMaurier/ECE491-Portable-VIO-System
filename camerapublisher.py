#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class MonoRepublisher:
    def __init__(self):
        self.bridge = CvBridge()
        in_topic  = rospy.get_param("~in",  "/libcamera_ros/image_raw")
        out_topic = rospy.get_param("~out", "/cam0/image_raw")

        self.pub = rospy.Publisher(out_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(in_topic, Image, self.cb, queue_size=1)

    def cb(self, msg: Image):
        enc = msg.encoding.lower()

        # Common case: RGB/BGR already decoded
        if enc in ["bgr8", "rgb8"]:
			
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=enc)
            
            mono = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if enc == "bgr8" else cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            
            out = self.bridge.cv2_to_imgmsg(mono, encoding="mono8")
            out.header = msg.header
            
            self.pub.publish(out)
            return

        # If your driver publishes yuv420 in one buffer, you can take the Y plane.
        # For yuv420p: first width*height bytes are Y.
        if "yuv420" in enc:
			
            w, h = msg.width, msg.height
            print("yuv420")
            data = np.frombuffer(msg.data, dtype=np.uint8)
            y = data[:w*h].reshape((h, w))  # luma
            out = self.bridge.cv2_to_imgmsg(y, encoding="mono8")
            out.header = msg.header
            self.pub.publish(out)
            return

        rospy.logwarn_throttle(5.0, f"Unsupported encoding for mono conversion: {msg.encoding}")

if __name__ == "__main__":
    rospy.init_node("mono_republisher")
    MonoRepublisher()
    rospy.spin()
