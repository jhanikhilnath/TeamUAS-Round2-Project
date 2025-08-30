import cv2
import numpy as np
import colorDetector as cd
import shapeDetector as sd
import os
import math


shape_list = {'star': 3, 'triangle': 2, 'square': 1}
color_list = {'red': 3, 'yellow': 2, 'green': 1}
center_color_list = {'pink': 3, 'blue': 4, 'grey': 2}
# img_list = os.listdir('./uas images')

# for i in img_list:
#   main(f'./uas images/{i}')

name = './uas images/1.png'


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_priority(patient):
    return shape_list[patient['shape']]*color_list[patient['color']]


def center_priority(center):
    sum = 0
    for patient in center['patients']:
        sum += calculate_priority(patient)

# higher the priority and the lesser distance will get the most priority


def calculate_score(center, patient):
    distance = distance(center['centre'], patient['centre'])
    return calculate_priority(patient)/distance


def maximise_center_score(centers, patients):
    sorted_patients = sorted(patients, key=lambda x: (
        calculate_priority(x), color_list[x['color']]))
    for patient in patients:
        best_center = None
        best_score = -math.inf
        for center in centers:
            if len(center['patients']) < center_color_list[center['color']]:
                score = calculate_score(center, patient)
                if score > best_score:
                    best_score = score
                    best_center = center
        if best_center:
            best_center['patients'].append(patient)


def main(path):
    img = cv2.imread(path)
    obj_list = sd.detect_shape(img)
    patient_list = []
    centre_list = []
    for i in obj_list:
        i['color'] = cd.detect_color(img, i['contour'])
        if not i['type']:
            patient_list.append(i)
        else:
            i['patients'] = []
            centre_list.append(i)

    # for i in obj_list:
    #     print(i['shape'], i['type'], i['color'], i['centre'])


main(name)
