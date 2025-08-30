import cv2
import numpy as np

import colorDetector as cd

img = cv2.imread('./uas images/1.png')

# Image Preprocessing

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3, 3), 0)

ret, thresh = cv2.threshold(blur, 130, 225, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)

contours, hierarchies = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Avoid Detection of whole photo frame as a separate contour by setting a max threshold and also a minimum area threshold
max_area_limit = img.shape[0] * img.shape[1]*0.4
min_area_limit = 25

# print(f'{len(contours)} contours found!!!')

# print(contours)

final_shape_list = []

# drawing overlay on the orignal image to easily test shape detection
for i, contour in enumerate(contours):
    # if i == 0:
    #     continue

    area = cv2.contourArea(contour)

    if area < min_area_limit:
        continue

    if area > max_area_limit:
        continue

    final_shape_list.append(contour)

    epsilon = 0.03*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(img, contour, -1, (0, 0, 255), 2)

    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + w/2)
    y_mid = int(y + h/2)

    coords = (x_mid, y_mid)
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX

    # cv2.putText(img, cd.detect_color(img, contour), coords, font, 1, color, 1)

    color_analysis = cd.detect_color(img, contour)

    if len(approx) == 3:
        cv2.putText(img, f"{color_analysis} Triangle",
                    coords, font, 1, color, 1)
    elif len(approx) == 4:
        cv2.putText(img, f"{color_analysis} square", coords, font, 1, color, 1)
    elif len(approx) == 10:
        cv2.putText(img, f"{color_analysis} star", coords, font, 1, color, 1)
    else:
        cv2.putText(img, f"{color_analysis} Circle", coords, font, 1, color, 1)

# Final Number of shapes
print(len(final_shape_list))

cv2.imshow('og', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
