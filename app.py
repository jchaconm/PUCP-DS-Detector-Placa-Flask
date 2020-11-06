import base64
import datetime
import uuid

import jwt
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, make_response
import pickle

from skimage.io import imread

from cv_operations import check_video
from predict import get_placa
from test_model import test
from tools import encode, delete_files_in_directory

import bcrypt
from flask import Flask,render_template,flash, redirect,url_for,session,logging,request
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user, current_user
from itsdangerous import URLSafeTimedSerializer
import json
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__) #Initialize the flask App
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

model = pickle.load(open('model.pkl', 'rb'))

app.config['SECRET_KEY'] = 'pucpdspf'
app.config['JWT_SECRET_KEY']="\xf9'\xe4p(\xa9\x12\x1a!\x94\x8d\x1c\x99l\xc7\xb7e\xc7c\x86\x02MJ\xa0"
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://tfikptad:iIVeftDYpaqg60wKy8uhTeVFH2Jb2STO@lallah.db.elephantsql.com:5432/tfikptad'
db = SQLAlchemy(app)


class historial(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    placa = db.Column(db.String)
    foto = db.Column(db.String)
    user_id = db.Column(db.String, db.ForeignKey('user.email'))

    def serialize(self):
        return {
            'placa': self.placa,
            'foto': self.foto,
        }

class user(db.Model):
    email = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)

    def is_active(self):
        """True, as all users are active."""
        return True

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        return self.email

    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    def is_anonymous(self):
        """False, as anonymous users aren't supported."""
        return False

    def encode_auth_token(self, user_id):
        """
        Generates the Auth Token
        :return: string
        """
        try:
            payload = {
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5000),
                'iat': datetime.datetime.utcnow(),
                'sub': user_id
            }
            return jwt.encode(
                payload,
                app.config.get('JWT_SECRET_KEY'),
                algorithm='HS256'
            )
        except Exception as e:
            return e

    @staticmethod
    def decode_auth_token(auth_token):
        """
        Decodes the auth token
        :param auth_token:
        :return: integer|string
        """
        try:
            payload = jwt.decode(auth_token, app.config.get('JWT_SECRET_KEY'))
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return 'Signature expired. Please log in again.'
        except jwt.InvalidTokenError:
            return 'Invalid token. Please log in again.'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/user-history',methods=['GET'])
def user_history():
    lst_predicciones = historial.query.filter_by(user_id="xyz")\
        .order_by(historial.id.desc()).all() #response_object["email"])
    for prediccion in lst_predicciones:
        car_image_output = imread("./result_img/" + prediccion.foto + ".jpg")
        base64img = encode(car_image_output)
        prediccion.foto = base64img
    return make_response(jsonify(results=[elem.serialize() for elem in lst_predicciones])), 200

@app.route('/history')
def history():
    #car_image_output = imread("./result_img/81251533702869185905593538477945253758.jpg")
    #base64img = encode(car_image_output)
    return render_template('history.html')#,image=base64img)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        response_object = validate_token();

        if response_object['status'] == 'success':
            try:
                video = request.files['video']
                plates_like_objects, image_binary, car_image = check_video(video, app.config['UPLOAD_FOLDER'])
                result_dict = get_placa(plates_like_objects, image_binary, model)
                prediction_id = uuid.uuid4()
                result_img = Image.fromarray(car_image).convert('RGB')
                # Guardar la imagen
                result_img.save("./result_img/" + str(prediction_id) +".jpg")
                # TODO guardar predicción , ruta de imagen en BD y usuario en tabla de BD,para listarlo luego en Historial
                nwPrediccion = historial(user_id=response_object['email'],placa=result_dict['placa'],
                                         foto=str(prediction_id))
                db.session.add(nwPrediccion)
                db.session.commit()

                base64img = encode(car_image)
                # limpiar carpeta output y uploads
                delete_files_in_directory(app.config['UPLOAD_FOLDER'])
                delete_files_in_directory('output')
                result_dict["img"] = base64img
                return make_response(jsonify(result_dict)), 200
            except Exception:
                return make_response(jsonify({"status":"failure"})), 400

        else:
            return make_response(jsonify(response_object)), 401
        #return render_template('index.html',image=base64img, prediction_text='Placa detectada es {}'.format(result_dict['placa']))


@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.json['email']
        passw = request.json['password']
        loginUser = user.query.get(email)
        if loginUser:
            if bcrypt.checkpw(passw.encode('utf-8'), loginUser.password.encode('utf-8')):
                auth_token = loginUser.encode_auth_token(loginUser.email)
                responseObject = {
                    'status': 'success',
                    'message': 'Successfully authenticated.',
                    'email': loginUser.email,
                    'token':  auth_token.decode()
                }
            else:
                responseObject = {
                    'status': 'failure',
                    'message': 'Wrong credentials.'
                }
            return make_response(jsonify(responseObject))

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.json['email']
        passw = request.json['password']
        hashed = bcrypt.hashpw(passw.encode(),  bcrypt.gensalt())
        register = user(email = email, password = hashed.decode('utf-8'))
        db.session.add(register)
        db.session.commit()
        responseObject = {
            'status': 'success',
            'message': 'Successfully registered.',
        }
        return make_response(jsonify(responseObject))
    return render_template("register.html")

@app.route("/home", methods=["GET"])
@login_required
def profile():
    return render_template('logged_in_page.html', email=current_user.email)

@app.route('/logout')
@login_required
def logout():
    return redirect(url_for("login"))

@app.route("/check", methods=["POST"])
def create_entry():
    req = request.get_json()
    print(req)
    res = make_response(jsonify({"message": "TEST PYTHON"}), 200)
    return res

def validate_token():
 # get the auth token
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            auth_token = auth_header.split(" ")[1]
        except IndexError:
            responseObject = {
                'status': 'fail',
                'message': 'Bearer token malformed.'
            }
            return responseObject
        if auth_token:
            resp = user.decode_auth_token(auth_token)
            if isinstance(resp, str):
                current_user = user.query.filter_by(email=resp).first()
                if current_user:
                    responseObject = {
                        'status': 'success',
                        'email': resp
                    }
                    return responseObject
            responseObject = {
                'status': 'fail',
                'message': resp
            }
            return responseObject
        else:
            responseObject = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return responseObject
    else:
        responseObject = {
            'status': 'fail',
            'message': 'No auth token provided.'
        }
        return responseObject

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)