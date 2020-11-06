
import pickle
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        dir_path = os.path.join(training_directory, each_letter)
        path, dirs, files = next(os.walk(dir_path))
        for each in files:
            image_path = os.path.join(dir_path,str(each))
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            #img_details = cv2.bitwise_not(img_details)

            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # the 2D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20*20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

# current_dir = os.path.dirname(os.path.realpath(__file__))
#
# training_dataset_dir = os.path.join(current_dir, 'train')
print('reading data')
training_dataset_dir = './dataset'
image_data, target_data = read_training_data(training_dataset_dir)
print('reading data completed')

# the kernel can be 'linear', 'poly' or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
X_train, X_test, y_train, y_test = train_test_split(image_data, target_data,test_size = 0.3,
                                                    random_state =0)
svc_model = SVC(C= 100, gamma= 0.001, kernel= 'linear')
print('training model')
# let's train the model with all the input data
svc_model.fit(image_data, target_data)
pred_train = svc_model.predict(X_train)
pred_test =svc_model.predict(X_test)
pickle.dump(svc_model, open('model.pkl','wb'))
