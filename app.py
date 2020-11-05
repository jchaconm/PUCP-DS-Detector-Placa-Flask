import base64
import uuid

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import pickle

from skimage.io import imread

from cv_operations import check_video
from predict import get_placa
from test_model import test
from tools import encode, delete_files_in_directory

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/history')
def history():
    car_image_output = imread("./result_img/81251533702869185905593538477945253758.jpg")
    base64img = encode(car_image_output)
    return render_template('test.html',image=base64img)

@app.route('/predict',methods=['POST'])
def predict():
    video = request.files['video']
    plates_like_objects,image_binary,car_image = check_video(video, app.config['UPLOAD_FOLDER'])
    result_dict = get_placa(plates_like_objects,image_binary,model)
    prediction_id = uuid.uuid4()
    result_img = Image.fromarray(car_image).convert('RGB')
    #Guardar la imagen
    result_img.save("./result_img/%d.jpg" % prediction_id)
    #TODO guardar predicción , ruta de imagen en BD y usuario en tabla de BD,para listarlo luego en Historial
    base64img = encode(car_image)
    #limpiar carpeta output y uploads
    delete_files_in_directory(app.config['UPLOAD_FOLDER'])
    delete_files_in_directory('output')
    return render_template('index.html',image=base64img, prediction_text='Placa detectada es {}'.format(result_dict['placa']))

if __name__ == "__main__":
    app.run(debug=True)