import cv2
import numpy as np

img = cv2.imread('./uas images/1.png')

blank = np.zeros(img.shape, dtype='uint8')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# edges = cv2.Canny(img, 100, 200)

# blur = cv2.GaussianBlur(gray, (5, 5), 0)
# blur = cv2.medianBlur(gray, 71)

edgesGRAY = cv2.Canny(gray, 100, 200)

ret, thresh = cv2.threshold(edgesGRAY, 100, 250, cv2.THRESH_BINARY_INV)

# cv2.imshow("thresh", thresh)

contours, hierarchies = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f'{len(contours)} contours found!!!')

for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 0.01*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(img, contour, -1, (0, 0, 255), 2)

    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + w/2)
    y_mid = int(y + h/2)

    coords = (x_mid, y_mid)
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX

    if len(approx) == 3:
        cv2.putText(img, "Triangle", coords, font, 1, color, 1)
    elif len(approx) == 4:
        cv2.putText(img, "square", coords, font, 1, color, 1)
    elif len(approx) == 10:
        cv2.putText(img, "star", coords, font, 1, color, 1)
    else:
        cv2.putText(img, "Circle", coords, font, 1, color, 1)

# cv2.imshow('contours', blank)

cv2.imshow('og', img)
# cv2.imshow('edges', edges)
# cv2.imshow('thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
