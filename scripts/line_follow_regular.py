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

# global variables
yaw_rate = Float32()

################### callback ###################

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
    print(f"Minimum y coordinate:", min_y)
    # the bottom of the image
    max_y = cv_image.shape[0]
    print(f"Maximum y coordinate:",max_y)
    # middle_line_y_start = min_y
    # middle_line_y_end = max_y

    poly_left = 0
    poly_right = 0
    poly_middle = 0

    if len(left_line_x) != 0 and len(left_line_y) != 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))
        left_line_x_start = int(poly_left(max_y))
        left_line_x_end = int(poly_left(min_y))
        print(f"Left line x coordinate:",left_line_x)
        print(f"Left line y coordinate:",left_line_y)
        print(f"Poly left:",poly_left)
        print(f"x coordinate Start point left:",left_line_x_start)
        print(f"x coordinate End point left:",left_line_x_end)
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
        middle_line_x_bottom = int(poly_middle(max_y))
        middle_line_x_top = int(poly_middle(max_y))
    else:
        middle_line_x_bottom = int(width/2)
        middle_line_x_top = int(width/3)

    side_lines= [[ [left_line_x_start, max_y, left_line_x_end, min_y], [right_line_x_start, max_y, right_line_x_end, min_y] ]]
    middle_line= [[ [middle_line_x_bottom, max_y, middle_line_x_top, min_y] ]]

    line_image = np.copy(cv_image) * 0
    copy_image = np.copy(cv_image) * 0

    for line in side_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (255, 0, 0), 10)

    for line in middle_line:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)
            cv2.line(copy_image, (x1, y1), (x2, int(y2)), (0, 0, 255), 10)

    lines_edges = cv2.addWeighted(cv_image, 0.9, line_image, 1, 0)
    middle_line_edge = cv2.addWeighted(cv_image, 0.9, copy_image, 1, 0)

    # convert the image to grayscale
    hsv = cv2.cvtColor(middle_line_edge, cv2.COLOR_BGR2HSV)
    thresh1 = cv2.inRange(hsv,np.array((0, 150, 150)), np.array((20, 255, 255)))
    thresh2 = cv2.inRange(hsv,np.array((150, 150, 150)), np.array((180, 255, 255)))
    thresh =  cv2.bitwise_or(thresh1, thresh2)

    # find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # initialize the variables for computing the centroid and finding the largest contour
    cx = 0
    cy = 0
    max_contour = []

    if len(contours) != 0:
        # find the largest contour by its area
        max_contour = max(contours, key = cv2.contourArea)

        # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        M = cv2.moments(max_contour)

        if M["m00"] != 0:
            # compute the x and y coordinates of the centroid
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print(f"cx:", cx)
    else:
        # rospy.loginfo(f"empty contours: {contours}")
        pass

    try:
        # draw the obtained contour lines (or the set of coordinates forming a line) on the original image
        cv2.drawContours(middle_line_edge, max_contour, -1, (0, 255, 0), 10)
    except UnboundLocalError:
        rospy.loginfo("max contour not found")

    # draw a circle at centroid (https://www.geeksforgeeks.org/python-opencv-cv2-circle-method)
    cv2.circle(middle_line_edge, (cx, cy), 8, (180, 0, 0), -1)  # -1 fill the circle
    
    pub_yaw_rate(middle_line_edge, cx, cy)

    cv2.imshow("CV Image", cv_image)
    cv2.imshow("Hough Lines", lines_edges)
    cv2.imshow("Middle Hough Lines", middle_line_edge)
    cv2.waitKey(3)
    #rate.sleep()


################### algorithms ###################

def pub_yaw_rate(cv_image, cx, cy):

    # get the dimensions of the image
    width = cv_image.shape[1]
    height = cv_image.shape[0]

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

    rospy.Subscriber("/camera_view", Image, image_callback)

    #rate = rospy.Rate(10)
    yaw_rate_pub = rospy.Publisher("yaw_rate", Float32, queue_size=1)

    try:
      rospy.spin()
    except rospy.ROSInterruptException:
      pass
