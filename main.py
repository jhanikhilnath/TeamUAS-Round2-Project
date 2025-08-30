import cv2
import numpy as np
import colorDetector as cd
import shapeDetector as sd
import os

# img_list = os.listdir('./uas images')

# for i in img_list:
#   main(f'./uas images/{i}')

name = './uas images/1.png'


def main(path):
    img = cv2.imread(path)
    obj_list = sd.detect_shape(img)
    for i in obj_list:
        i['color'] = cd.detect_color(img, i['contour'])

    for i in obj_list:
        print(i['shape'], i['type'], i['color'], i['centre'])


main(name)
