import cv2
import os


def id_draw(img, i, list_of_contours):
    font = cv2.FONT_HERSHEY_DUPLEX
    color = (0, 0, 0)

    for cntr in list_of_contours:
        cv2.putText(img, cntr['id'],
                    cntr['centre'], font, 1, color, 2)

    image_name = f"{i}_id.png"
    os.makedirs('output_folder/id_marked', exist_ok=True)
    output_path = os.path.join('./output_folder/id_marked', image_name)

    cv2.imwrite(output_path, img)
