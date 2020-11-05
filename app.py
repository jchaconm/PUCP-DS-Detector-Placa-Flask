import base64

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from cv_operations import check_video
from predict import get_placa
from test_model import test
from tools import encode

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    '''
    video = request.files['video']
    plates_like_objects,image_binary,car_image = check_video(video, app.config['UPLOAD_FOLDER']);
    result_dict = get_placa(plates_like_objects,image_binary,model);
    base64img = encode(car_image);
    return render_template('index.html',image=base64img, prediction_text='Placa detectada es {}'.format(result_dict['placa']))

if __name__ == "__main__":
    app.run(debug=True)