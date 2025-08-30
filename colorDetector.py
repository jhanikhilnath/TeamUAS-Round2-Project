import cv2
import numpy as np


def detect_color(img, cntr):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, [cntr], 255)

    # Compute all the colored masks that we need to identify
    # red_mask = cv2.bitwise_or(cv2.inRange(
    #     hsv, (0, 50, 50), (10, 255, 255)), cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)))
    red_mask = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (30, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    grey_mask = cv2.inRange(hsv, (0, 0, 50), (179, 30, 200))

    # Calculate Probability of color
    red_prob = cv2.countNonZero(cv2.bitwise_and(red_mask, mask))
    yellow_prob = cv2.countNonZero(cv2.bitwise_and(yellow_mask, mask))
    green_prob = cv2.countNonZero(cv2.bitwise_and(green_mask, mask))
    pink_prob = cv2.countNonZero(cv2.bitwise_and(pink_mask, mask))
    grey_prob = cv2.countNonZero(cv2.bitwise_and(grey_mask, mask))
    blue_prob = cv2.countNonZero(cv2.bitwise_and(blue_mask, mask))

    color_prob = {'red': red_prob, 'yellow': yellow_prob, "green": green_prob,
                  "pink": pink_prob, "grey": grey_prob, "blue": blue_prob}

    return max(color_prob, key=color_prob.get)
