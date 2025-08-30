import cv2
import numpy as np


def detect_shape(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(blur, 130, 225, cv2.THRESH_BINARY)

    contours, hierarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Avoid Detection of whole photo frame as a separate contour by setting a max threshold and also a minimum area threshold
    max_area_limit = img.shape[0] * img.shape[1]*0.4
    min_area_limit = 25

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

        epsilon = 0.03*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + w/2)
        y_mid = int(y + h/2)

        coords = (x_mid, y_mid)

        # stars representing children, squares representing adults, and triangles representing elderly individuals.

        if len(approx) == 3:
            shape, obj_type = 'triangle', 0
        elif len(approx) == 4:
            shape, obj_type = 'square', 0
        elif len(approx) == 10:
            shape, obj_type = 'star', 0
        else:
            shape, obj_type = 'circle', 1

        # type: 0->person, 1->rescue centre
        contour_info = {"shape": shape, "type": obj_type,
                        "centre": coords, "contour": contour}

        final_shape_list.append(contour_info)

    # Final Number of shapes
    return final_shape_list
