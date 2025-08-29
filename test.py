import cv2
import numpy as np

img = cv2.imread('./uas images/1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img, 100, 200)

edgesGRAY = cv2.Canny(gray, 100, 200)


cv2.imshow('og', img)
cv2.imshow('edges', edges)
cv2.imshow('edgesG', edgesGRAY)

cv2.waitKey(0)
cv2.destroyAllWindows()
