import cv2
import numpy as np
import os


def segment_ocean_land(img, i):

    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Segment ocean and land using color thresholds
    ocean_mask = cv2.inRange(hsv, (100, 50, 50), (140, 200, 200))
    land_mask = cv2.inRange(hsv, (40, 40, 40), (70, 210, 170))
    kernel = np.ones((5, 5), np.uint8)
    ocean_mask = cv2.morphologyEx(ocean_mask, cv2.MORPH_CLOSE, kernel)
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel)

    # Create an empty output image
    segmented = np.zeros_like(img)
    # Assign colors
    segmented[ocean_mask > 0] = [255, 100, 100]
    segmented[land_mask > 0] = [50, 255, 255]

    # Prepare the output path
    image_name = f"{i}_output.png"
    os.makedirs('output_folder', exist_ok=True)
    output_path = os.path.join('./output_folder', image_name)

    # Save the result (segmented) image
    cv2.imwrite(output_path, segmented)
    # print(f"Segmented image saved to {output_path}")


def detect_color(img, cntr, cntr_type):
    # Convert image to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask and fill it with only the contour passed to the function
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, [cntr], 255)

    # Compute all the colored masks that we need to identify
    # red_mask = cv2.bitwise_or(cv2.inRange(
    #     hsv, (0, 50, 50), (10, 255, 255)), cv2.inRange(hsv, (170, 50, 50), (180, 255, 255)))
    if not cntr_type:
        red_mask = cv2.inRange(hsv, (170, 50, 50), (180, 255, 255))
        yellow_mask = cv2.inRange(hsv, (15, 50, 50), (30, 255, 255))
        green_mask = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))
    else:
        pink_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        grey_mask = cv2.inRange(hsv, (0, 0, 50), (179, 30, 200))

    # Calculate Probability of color
    if not cntr_type:
        red_prob = cv2.countNonZero(cv2.bitwise_and(red_mask, mask))
        yellow_prob = cv2.countNonZero(cv2.bitwise_and(yellow_mask, mask))
        green_prob = cv2.countNonZero(cv2.bitwise_and(green_mask, mask))
    else:
        pink_prob = cv2.countNonZero(cv2.bitwise_and(pink_mask, mask))
        grey_prob = cv2.countNonZero(cv2.bitwise_and(grey_mask, mask))
        blue_prob = cv2.countNonZero(cv2.bitwise_and(blue_mask, mask))

    if not cntr_type:
        color_prob = {'red': red_prob,
                      'yellow': yellow_prob, "green": green_prob}
    else:
        color_prob = {"pink": pink_prob, "grey": grey_prob, "blue": blue_prob}

    # Return the color which has the maximum probability according to the algorithm
    return max(color_prob, key=color_prob.get)
