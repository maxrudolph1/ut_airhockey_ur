# import cv2
# import rospy
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image

# class Segmenter():
#     def __init__(self, args):
#         self.args = args
#         rospy.init_node("fg_bg_segmenter")
#         self.sub = rospy.Subscriber(args.img_topic, Image, self.cb)

#         self.title_window = 'Foreground-Background HSV Segmenter'

#         self.lower_hue = 0
#         self.upper_hue = 255
#         self.lower_sat = 0
#         self.upper_sat = 255
#         self.lower_val = 0
#         self.upper_val = 255

#         self.bgr = None
#         self.hsv = None
#         self.dst = None
#         self.bridge = CvBridge()

#         cv2.namedWindow(self.title_window)

#         cv2.createTrackbar('lower hue', self.title_window, 0, 255, self.on_lower_h)
#         cv2.createTrackbar('upper hue', self.title_window, 255, 255, self.on_upper_h)
#         cv2.createTrackbar('lower sat', self.title_window, 0, 255, self.on_lower_s)
#         cv2.createTrackbar('upper sat', self.title_window, 255, 255, self.on_upper_s)
#         cv2.createTrackbar('lower val', self.title_window, 0, 255, self.on_lower_v)
#         cv2.createTrackbar('upper val', self.title_window, 255, 255, self.on_upper_v)

#     def update(self):
#         lower = np.array([self.lower_hue, self.lower_sat, self.lower_val])
#         upper = np.array([self.upper_hue, self.upper_sat, self.upper_val])
#         mask = cv2.inRange(self.hsv, lower, upper)
#         # kernel = np.ones((9,9),np.uint8)
#         # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#         self.dst = cv2.bitwise_and(np.stack([mask, mask, mask], axis=2), self.bgr)
#         print("lower HSV: {}, upper HSV: {}".format(lower, upper))

#     def on_lower_h(self, val):
#         self.lower_hue = val
#         self.update()

#     def on_upper_h(self, val):
#         self.upper_hue = val
#         self.update()

#     def on_lower_s(self, val):
#         self.lower_sat = val
#         self.update()

#     def on_upper_s(self, val):
#         self.upper_sat = val
#         self.update()

#     def on_lower_v(self, val):
#         self.lower_val = val
#         self.update()

#     def on_upper_v(self, val):
#         self.upper_val = val
#         self.update()

#     def cb(self, msg):
#         im = self.bridge.imgmsg_to_cv2(msg)
#         self.bgr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         self.hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
#     def spin(self):
#         while not rospy.is_shutdown():
#             if self.bgr is not None:
#                 self.update()
#             if self.dst is None:
#                 rospy.sleep(0.1)
#             else:
#                 cv2.imshow(self.title_window, self.dst)
#                 cv2.waitKey(30)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_topic', help='ROS topic to subscribe to for RGB image', default='/camera/color/image_rect_color')
#     args, _ = parser.parse_known_args()
#     # path = "/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_rect/ms_two_side_horz_0_gray.png"
#     # path = "/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_lblue/towel_train_9_high_gray.png"
#     # path = "/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_shirt/open_2side_high_gray.png"
#     path = "/home/tweng/ws1/src/bimanual_folding/bimanual_ros/scripts/goals/real_high/open_2side_high_gray.png"

#     s = Segmenter(args)
#     s.spin()

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class Segmenter():
    def __init__(self, args):
        self.args = args

        self.title_window = 'Foreground-Background HSV Segmenter'

        self.lower_hue = 0
        self.upper_hue = 255
        self.lower_sat = 0
        self.upper_sat = 255
        self.lower_val = 0
        self.upper_val = 255

        self.bgr = None
        self.hsv = None
        self.dst = None
        self.bridge = CvBridge()

        cv2.namedWindow(self.title_window)

        cv2.createTrackbar('lower hue', self.title_window, 0, 255, self.on_lower_h)
        cv2.createTrackbar('upper hue', self.title_window, 255, 255, self.on_upper_h)
        cv2.createTrackbar('lower sat', self.title_window, 0, 255, self.on_lower_s)
        cv2.createTrackbar('upper sat', self.title_window, 255, 255, self.on_upper_s)
        cv2.createTrackbar('lower val', self.title_window, 0, 255, self.on_lower_v)
        cv2.createTrackbar('upper val', self.title_window, 255, 255, self.on_upper_v)

    def update(self):
        lower = np.array([self.lower_hue, self.lower_sat, self.lower_val])
        upper = np.array([self.upper_hue, self.upper_sat, self.upper_val])
        mask = cv2.inRange(self.hsv, lower, upper)
        self.dst = cv2.bitwise_and(np.stack([mask, mask, mask], axis=2), self.bgr)
        print("lower HSV: {}, upper HSV: {}".format(lower, upper))

    def on_lower_h(self, val):
        self.lower_hue = val
        self.update()

    def on_upper_h(self, val):
        self.upper_hue = val
        self.update()

    def on_lower_s(self, val):
        self.lower_sat = val
        self.update()

    def on_upper_s(self, val):
        self.upper_sat = val
        self.update()

    def on_lower_v(self, val):
        self.lower_val = val
        self.update()

    def on_upper_v(self, val):
        self.upper_val = val
        self.update()

    def read_image(self, path):
        im = cv2.imread(path)
        self.bgr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    def spin(self):
        while True:
            if self.bgr is not None:
                self.update()
            if self.dst is None:
                cv2.waitKey(100)
            else:
                cv2.imshow(self.title_window, self.dst)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', help='Path to the image file', required=True)
    args, _ = parser.parse_known_args()

    s = Segmenter(args)
    s.read_image(args.image_path)
    s.spin()
