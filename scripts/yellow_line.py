#!/usr/bin/env python3

# yellow line detection node

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32, Float32
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_sim.cfg import YellowLineConfig

# global variables
yellow_msg = Bool()

previous_time = 0
yellow_frames = 0

publish_once = True

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def image_callback(camera_image):
    global previous_time, yellow_frames, publish_once

    try:
        cv_image = CvBridge().imgmsg_to_cv2(camera_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)

    # convert image to HLS colorspace
    hls_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)

    height = hls_image.shape[0]
    width = hls_image.shape[1]
    width = int(width / 4)

    # crop the image
    roi_image = hls_image[:, width:width * 3]

    # specify the hls values for yellow
    lower_bounds = np.uint8([10, RC.light_low, RC.sat_low])
    upper_bounds = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(roi_image, lower_bounds, upper_bounds)

    contours, _ = cv2.findContours (yellow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # initialize the variables for computing the centroid and finding the largest contour
    max_area = 0
    max_contour = []

    if len(contours) != 0:
        # find the largest contour by its area
        max_contour = max(contours, key = cv2.contourArea)
        max_area = cv2.contourArea(max_contour)

    # draw the obtained contour lines(or the set of coordinates forming a line) on the original image
    cv2.drawContours(roi_image, max_contour, -1, (0, 0, 255), 8)

    # this number should be determined based on speed
    num_frames = 3

    # detect yellow for a continuous number of frames
    if (max_area > 50) and (yellow_frames < num_frames):
        yellow_msg.data = False
        yellow_frames += 1
    elif (yellow_frames == num_frames) and publish_once:
        yellow_msg.data = True
        publish_once = False
        yellow_frames = 0
    elif (max_area == 0) and (yellow_frames == 0) and not publish_once:
        publish_once = True
    else:
        yellow_msg.data = False
        yellow_frames = 0

    yellow_msg_pub.publish(yellow_msg)

    # concatenate the roi images to show in a single window
    # the shape of the images must have the same length: len(image.shape)
    yellow_mask_channel = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    concatenated_roi_image = cv2.hconcat([roi_image, yellow_mask_channel])

    cv2.putText(concatenated_roi_image, f"MAX_AREA: {max_area}", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(concatenated_roi_image, f"DETECTED: {yellow_msg.data}", (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 100), 2)
    cv2.imshow("Yellow Line", concatenated_roi_image)
    cv2.waitKey(3)

################### main ###################

if __name__ == "__main__":
    rospy.init_node("yellow_line", anonymous=True)

    rospy.Subscriber("/camera_view", Image, image_callback)

    yellow_msg_pub = rospy.Publisher("/yellow_line_detected", Bool, queue_size=1)

    dynamic_reconfigure_server = Server(YellowLineConfig, dynamic_reconfigure_callback)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
