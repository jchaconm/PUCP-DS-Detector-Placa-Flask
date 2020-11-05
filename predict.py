import cv2
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
import base64
import json
from PIL import Image
import numpy as np
import io
from tools import encode


def callALPR(image):
   base64image = encode(image)
   SECRET_KEY = 'sk_d6c76f5e23014eff289f1313'
   url = 'https://api.openalpr.com/v3/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
   r = requests.post(url, data = base64image, verify=False)
   output_dict = r.json()
   return {"placa": output_dict['results'][0]['plate'],
            "flag_alpr" : True}

def get_placa(plate_like_objects,binary_car_image, svc_model):
    if (len(plate_like_objects) == 0):
        return callALPR(binary_car_image)
        # The invert was done so as to convert the black pixel to white pixel and vice versa
    license_plate = np.invert(plate_like_objects[0])
    # license_plate = plate_like_objects[0]
    labelled_plate = measure.label(license_plate)
    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some
    character_dimensions = (0.35 * license_plate.shape[0], 0.60 * license_plate.shape[0], 0.05 * license_plate.shape[1],
                            0.15 * license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter = 0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
            roi = cv2.resize(np.float32(roi), (20, 20), interpolation=cv2.INTER_CUBIC)
            # draw a red bordered rectangle over the character.
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
                                            linewidth=2, fill=False)
            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)
            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)

    classification_result = []

    for ix, each_character in enumerate(characters):
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        result = svc_model.predict(each_character)
        classification_result.append(result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    return {"placa": rightplate_string,
            "flag_alpr": False}