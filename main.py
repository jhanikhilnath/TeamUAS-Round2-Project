import cv2
import numpy as np
import colorDetector as cd
import shapeDetector as sd
import os
import math


shape_list = {'star': 3, 'triangle': 2, 'square': 1}
color_list = {'red': 3, 'yellow': 2, 'green': 1}
center_color_list = {'pink': 3, 'blue': 4, 'grey': 2}
img_list = os.listdir('./uas images')

for i in img_list:
    print(int(i[:-4]))

name = './uas images/2.png'


def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_priority(patient):
    print(patient['color'])
    return shape_list[patient['shape']]*color_list[patient['color']]


def center_priority(center):
    sum = 0
    for patient in center['patients']:
        sum += calculate_priority(patient)

# higher the priority and the lesser distance will get the most priority


def calculate_score(center, patient):
    point_distance = distance(center['centre'], patient['centre'])
    return calculate_priority(patient)/point_distance


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

# Distance Matrix
# p1 - [c1, c2, c3]
# p2 - [c1, c2, c3]


def distance_matrix(centers, patients):
    matrix = []
    for patient in patients:
        p = []
        for center in centers:
            p.append(distance(patient['centre'], center['centre']))
        matrix.append(p)

    return np.array(matrix, dtype=np.float32)


def center_patient_list_gen(centers):
    blue_center = []
    pink_center = []
    grey_center = []
    for center in centers:
        x = []
        for patient in center['patients']:
            x.append(
                [shape_list[patient['shape']], color_list[patient['color']]])

        if center['color'] == 'pink':
            pink_center = x
        elif center['color'] == 'blue':
            blue_center = x
        else:
            grey_center = x

    # Return Order is [blue, pink, grey]
    return [blue_center, pink_center, grey_center]


def center_score_calc(center):
    center_sum = 0
    for i in center:
        center_sum += i[0]*i[1]
    return center_sum


def main(path):
    img = cv2.imread(path)
    obj_list = sd.detect_shape(img)
    patient_list = []
    centre_list = []
    for i in obj_list:
        i['color'] = cd.detect_color(img, i['contour'], i['type'])
        if not i['type']:
            patient_list.append(i)
        else:
            i['patients'] = []
            centre_list.append(i)

    maximise_center_score(centre_list, patient_list)
    print(distance_matrix(centre_list, patient_list))
    center_patient_list = center_patient_list_gen(centre_list)
    print(center_patient_list)
    center_score_list = []
    for i in center_patient_list:
        center_score_list.append(center_score_calc(i))

    print(center_score_list)
    cd.segment_ocean_land(img, 1)

    final_image_score = sum(center_score_list) / len(patient_list)
    print(final_image_score)


main(name)
