import uuid

from skimage.io import imread
from skimage.filters import threshold_otsu
import imutils
import cv2
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import os
import shutil

from werkzeug.utils import secure_filename



def check_video(video,upload_folder):
    if os.path.exists('output'):
        shutil.rmtree('output')
    os.makedirs('output')
    filename = secure_filename(video.filename)
    video.save(os.path.join(upload_folder, filename))
    cap = cv2.VideoCapture(os.path.join(upload_folder, filename))
    # cap = cv2.VideoCapture(0)
    count = 0

    while cap.isOpened():
        ret,frame = cap.read()
        if ret == True:
            #cv2_imshow('window-name',frame)
            cv2.imwrite("./output/frame%d.jpg" % count, frame)
            count = count + 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    # car image -> grayscale image -> binary image
    car_image_output = imread("./output/frame%d.jpg"%(count-1))
    car_image_output = imutils.rotate(car_image_output, 270)
    car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
    car_image = imutils.rotate(car_image, 270)


    # the next line is not compulsory however, a grey scale pixel
    # in skimage ranges between 0 & 1. multiplying it with 255
    # will make it range between 0 & 255 (something we can relate better with

    gray_car_image = car_image * 255

    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    # CCA (finding connected regions) of binary image
    # this gets all the connected regions and groups them together
    label_image = measure.label(binary_car_image)

    # getting the maximum width, height and minimum width and height that a license plate can be
    plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
    plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions
    plate_objects_cordinates = []
    plate_like_objects = []


    flag =0
    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        # print(region)
        if region.area < 50:
            #if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col
        # ensuring that the region identified satisfies the condition of a typical license plate
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            flag = 1
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                            max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                          linewidth=2, fill=False)

    if(flag==0):
        min_height, max_height, min_width, max_width = plate_dimensions2
        plate_objects_cordinates = []
        plate_like_objects = []
        # regionprops creates a list of properties of all the labelled regions
        for region in regionprops(label_image):
            if region.area < 50:
                #if the region is so small then it's likely not a license plate
                continue
                # the bounding box coordinates
            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            # ensuring that the region identified satisfies the condition of a typical license plate
            if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
                plate_like_objects.append(binary_car_image[min_row:max_row,
                                          min_col:max_col])
                plate_objects_cordinates.append((min_row, min_col,
                                                max_row, max_col))
                rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                              linewidth=2, fill=False)

        return (plate_like_objects,binary_car_image,car_image_output)
