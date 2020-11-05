from skimage import data, io
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_otsu
import cv2

def test(model):
    img_test = imread("./dataset/0/0_0.jpg", as_gray=True) #probar con imágenes de carácteres
    binary_image = img_test < threshold_otsu(img_test)
    flat_bin_image = binary_image.reshape(1,-1)
    io.imshow(img_test)
    plt.show()
    print(flat_bin_image.shape)
    return model.predict(flat_bin_image)