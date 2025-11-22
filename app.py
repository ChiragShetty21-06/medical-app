from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# ------------------------- Load ML Models -------------------------
covid_model = load_model('models/covid.h5')
braintumor_model = load_model('models/braintumor.h5')
alzheimer_model = load_model('models/alzheimer_model.h5')
diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
pneumonia_model = load_model('models/pneumonia_model.h5')
breastcancer_model = joblib.load('models/cancer_model.pkl')

# ------------------------- Flask Config -------------------------
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ------------------ Brain Tumor Helper Functions ------------------

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

# ------------------------- Routes -------------------------

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')

@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')

# ------------------------- Result Routes -------------------------

@app.route('/resultc', methods=['POST'])
def resultc():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    email = request.form['email']
    phone = request.form['phone']
    gender = request.form['gender']
    age = request.form['age']
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image uploaded successfully')
        img = cv2.imread('static/uploads/' + filename)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3) / 255.0
        pred = covid_model.predict(img)
        pred = 1 if pred >= 0.5 else 0

        return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname,
                               age=age, r=pred, gender=gender)

    flash('Invalid file format')
    return redirect(request.url)


@app.route('/resultbt', methods=['POST'])
def resultbt():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    gender = request.form['gender']
    age = request.form['age']
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread('static/uploads/' + filename)
        img = crop_imgs([img])
        img = img.reshape(img.shape[1:])
        img = preprocess_imgs([img], (224, 224))
        pred = braintumor_model.predict(img)
        pred = 1 if pred >= 0.5 else 0

        return render_template('resultbt.html', filename=filename, fn=firstname, ln=lastname,
                               age=age, r=pred, gender=gender)

    flash('Invalid file format')
    return redirect(request.url)


@app.route('/resultd', methods=['POST'])
def resultd():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    gender = request.form['gender']
    age = request.form['age']

    vals = [
        request.form['pregnancies'],
        request.form['glucose'],
        request.form['bloodpressure'],
        request.form['skin'],
        request.form['insulin'],
        request.form['bmi'],
        request.form['diabetespedigree'],
        request.form['age'],
    ]

    pred = diabetes_model.predict([vals])

    return render_template('resultd.html', fn=firstname, ln=lastname,
                           age=age, r=pred, gender=gender)


@app.route('/resultbc', methods=['POST'])
def resultbc():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    gender = request.form['gender']
    age = request.form['age']

    vals = np.array([
        request.form['concave_points_mean'],
        request.form['area_mean'],
        request.form['radius_mean'],
        request.form['perimeter_mean'],
        request.form['concavity_mean'],
    ]).reshape(1, -1)

    pred = breastcancer_model.predict(vals)

    return render_template('resultbc.html', fn=firstname, ln=lastname,
                           age=age, r=pred, gender=gender)


@app.route('/resultp', methods=['POST'])
def resultp():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    gender = request.form['gender']
    age = request.form['age']
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread('static/uploads/' + filename)
        img = cv2.resize(img, (150, 150))
        img = img.reshape(1, 150, 150, 3) / 255.0
        pred = pneumonia_model.predict(img)
        pred = 1 if pred >= 0.5 else 0

        return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname,
                               age=age, r=pred, gender=gender)

    flash('Invalid file format')
    return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    firstname = request.form['firstname']
    lastname = request.form['lastname']
    gender = request.form['gender']
    age = float(request.form['age'])

    vals = np.array([
        float(request.form['nmv']),
        float(request.form['tcp']),
        float(request.form['eia']),
        float(request.form['thal']),
        float(request.form['op']),
        float(request.form['mhra']),
        age
    ]).reshape(1, -1)

    pred = heart_model.predict(vals)

    return render_template('resulth.html', fn=firstname, ln=lastname,
                           age=age, r=pred, gender=gender)


# ------------------------- No Cache -------------------------
@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

# ------------------------- Render Fix -------------------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
