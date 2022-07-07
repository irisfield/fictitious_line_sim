#!/usr/bin/env python3
# https://www.youtube.com/watch?v=AbqErp4ZGgU
# https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
# https://towardsdatascience.com/finding-driving-lane-line-live-with-opencv-f17c266f15db


import cv2
import math
import rospy
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from fictitious_line_sim.cfg import ControlUnitConfig

# global variables
yaw_rate = Float32()

################### callback ###################

def dynamic_reconfigure_callback(config, level):
    global RC
    RC = config
    return config

def image_callback(camera_image):

    try:
        # convert camera_image into an opencv-compatible image
        cv_image = CvBridge().imgmsg_to_cv2(camera_image, "bgr8")
    except CvBridgeError:
        print(CvBridgeError)
    width = cv_image.shape[0]
    height = cv_image.shape[1]

    # resize the image
    cv_image = cv2.resize(cv_image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

    #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #filtered_image = cv2.medianBlur(gray_image, 15)
    #canny = cv2.Canny(filtered_image, 100, 200)

    # apply filters to the image
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_image, 200, 255)
    ###################################################################################################
    lines = cv2.HoughLinesP(canny, rho=6, theta=(np.pi / 180),
                            threshold=15, lines=np.array([]), minLineLength=20, maxLineGap=30)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
            if (math.fabs(slope) < 0.5):
                continue
            if slope < 0:
                # if the slope is negative, left line
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    else:
        # print("\n\n LINES IS EMPTY \n\n")
        pass

    # just below the horizon
    min_y = cv_image.shape[0] * (3 / 5)
    # the bottom of the image
    max_y = cv_image.shape[0]

    poly_left = 0
    poly_right = 0
    poly_middle = 0

    if len(left_line_x) != 0 and len(left_line_y) != 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_line_x_start = int(poly_left(max_y))
        left_line_x_end = int(poly_left(min_y))
        # print(f"Left line x coordinate:",left_line_x)
        # print(f"Left line y coordinate:",left_line_y)
        # print(f"Poly left:",poly_left)
        # print(f"x coordinate Start point left:",left_line_x_start)
        # print(f"x coordinate End point left:",left_line_x_end)
    else:
        left_line_x_start = int(width/20)
        left_line_x_end = int(width/17)


    if len(right_line_x) != 0 and len(right_line_y) != 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))
        right_line_x_start = int(poly_right(max_y))
        right_line_x_end = int(poly_right(min_y))
    else:
        right_line_x_start = int(width/1)
        right_line_x_end = int(width/2)

    if poly_left != 0 and poly_right != 0:
        poly_middle = np.poly1d(np.polyfit(poly_left, poly_right, deg=1))
        middle_line_x_start = int(poly_middle(max_y))
        middle_line_x_end = int(poly_middle(max_y))
    else:
        middle_line_x_start = int(width/2)
        middle_line_x_end = int(width/3)

    side_lines= [[ [left_line_x_start, max_y, left_line_x_end, min_y], [right_line_x_start, max_y, right_line_x_end, min_y] ]]
    middle_line= [[ [middle_line_x_start, max_y, middle_line_x_end, min_y] ]]

    line_image = np.copy(cv_image) * 0
    copy_image = np.copy(cv_image) * 0

    for line in side_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (255, 0, 0), 10)

    for line in middle_line:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)
            cv2.line(copy_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)
            cx = abs(x2)
            print(f"cx:", cx)

    lines_edges = cv2.addWeighted(cv_image, 0.9, line_image, 1, 0)
    middle_line_edge = cv2.addWeighted(cv_image, 0.9, copy_image, 1, 0)

    ###################################################################################################

    # get the dimension of the image
    height, width = cv_image.shape[0], cv_image.shape[1]

    # offset the x position of the vehicle to follow the lane
    # cx -= 170
    (rows,cols,channels) = cv_image.shape

    # get the dimension of the image
    drive_2_follow_line(cv_image, cx-10, cols)

    #pub_yaw_rate(cx, height, width)


    cv2.imshow("CV Image", cv_image)
    cv2.imshow("Hough Lines", lines_edges)
    cv2.imshow("Middle Hough Lines", middle_line_edge)
    cv2.waitKey(3)
    #rate.sleep()


################### algorithms ###################
def drive_2_follow_line(cv_image, cx, cols): # algorithm 1
    mid = cols / 2
    print(f'mid:', mid)
    if cx > mid+5:
      #cv2.putText(cv_image,f"Turn Right", (10,rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
      yaw_rate.data = -0.1
    elif cx < mid-5:
      #cv2.putText(cv_image,f"Turn Left", (10,rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
      yaw_rate.data = 0.1
    else:
      #cv2.putText(cv_image,f"Go Stright", (10,rows-10), font, 1,(125,125,125),2,cv2.LINE_AA)
      yaw_rate.data = 0.0

    yaw_rate_pub.publish(yaw_rate)

    return

def pub_yaw_rate(cx, width, height):

    # compute the coordinates for the center the vehicle's camera view
    camera_center_y = (height / 2)
    camera_center_x = (width / 2)

    # compute the difference between the x and y coordinates of the centroid and the vehicle's camera center
    center_error = cx - camera_center_x

    # In simulation:
    #       less than 3.0 - deviates a little inward when turning
    #                 3.0 - follows the line exactly
    #       more than 3.0 - deviates a little outward when turning
    correction = 3.0 * camera_center_y

    # compute the yaw rate proportion to the difference between centroid and camera center
    angular_z = float(center_error / correction)

    if cx > camera_center_x:
        # angular.z is negative; left turn
        yaw_rate.data = -abs(angular_z)
    elif cx < camera_center_x:
        # angular.z is positive; right turn
        yaw_rate.data = abs(angular_z)
    else:
        # keep going straight
        yaw_rate.data = 0.0

    yaw_rate_pub.publish(yaw_rate)

    return

################### main ###################

if __name__ == "__main__":

    rospy.init_node("follow_line", anonymous=True)

    rospy.Subscriber("/camera/image_raw", Image, image_callback)

    #rate = rospy.Rate(10)
    yaw_rate_pub = rospy.Publisher("yaw_rate", Float32, queue_size=1)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
