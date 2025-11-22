from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# ------------------------- Load ML Models -------------------------

# Heavy CNN model (keep)
braintumor_model = load_model('models/braintumor.h5')

# Lightweight ML models
diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
breastcancer_model = joblib.load('models/cancer_model.pkl')

# ------------------------- Flask Config -------------------------

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# ------------------ Brain Tumor Helper Functions ------------------


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG16 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[
            extTop[1] - ADD_PIXELS : extBot[1] + ADD_PIXELS,
            extLeft[0] - ADD_PIXELS : extRight[0] + ADD_PIXELS,
        ].copy()
        set_new.append(new_img)

    return np.array(set_new)


# ------------------------- Page Routes -------------------------


@app.route("/")
def home():
    return render_template("homepage.html")


@app.route("/breastcancer")
def breast_cancer():
    return render_template("breastcancer.html")


@app.route("/braintumor")
def brain_tumor():
    return render_template("braintumor.html")


@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")


@app.route("/heartdisease")
def heartdisease():
    return render_template("heartdisease.html")


# ------------------------- Result Routes -------------------------


@app.route("/resultbt", methods=["POST"])
def resultbt():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]
    firstname = request.form.get("firstname", "")
    lastname = request.form.get("lastname", "")
    gender = request.form.get("gender", "")
    age = request.form.get("age", "")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        img = cv2.imread(save_path)
        img = crop_imgs([img])
        img = img.reshape(img.shape[1:])
        img = preprocess_imgs([img], (224, 224))
        pred = braintumor_model.predict(img)
        pred = 1 if pred >= 0.5 else 0

        return render_template(
            "resultbt.html",
            filename=filename,
            fn=firstname,
            ln=lastname,
            age=age,
            r=pred,
            gender=gender,
        )

    flash("Allowed image types are - png, jpg, jpeg")
    return redirect(request.url)


@app.route("/resultd", methods=["POST"])
def resultd():
    firstname = request.form.get("firstname", "")
    lastname = request.form.get("lastname", "")
    gender = request.form.get("gender", "")
    age = request.form.get("age", "")

    try:
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        bloodpressure = float(request.form["bloodpressure"])
        skinthickness = float(request.form["skin"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabetespedigree = float(request.form["diabetespedigree"])
        age_val = float(request.form["age"])
    except (KeyError, ValueError):
        flash("Invalid input values")
        return redirect(request.url)

    features = [
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        diabetespedigree,
        age_val,
    ]

    pred = diabetes_model.predict([features])[0]

    return render_template(
        "resultd.html",
        fn=firstname,
        ln=lastname,
        age=age,
        r=pred,
        gender=gender,
    )


@app.route("/resultbc", methods=["POST"])
def resultbc():
    firstname = request.form.get("firstname", "")
    lastname = request.form.get("lastname", "")
    gender = request.form.get("gender", "")
    age = request.form.get("age", "")

    try:
        cpm = float(request.form["concave_points_mean"])
        am = float(request.form["area_mean"])
        rm = float(request.form["radius_mean"])
        pm = float(request.form["perimeter_mean"])
        cm = float(request.form["concavity_mean"])
    except (KeyError, ValueError):
        flash("Invalid input values")
        return redirect(request.url)

    features = np.array([cpm, am, rm, pm, cm]).reshape(1, -1)
    pred = breastcancer_model.predict(features)[0]

    return render_template(
        "resultbc.html",
        fn=firstname,
        ln=lastname,
        age=age,
        r=pred,
        gender=gender,
    )


@app.route("/resulth", methods=["POST"])
def resulth():
    firstname = request.form.get("firstname", "")
    lastname = request.form.get("lastname", "")
    gender = request.form.get("gender", "")

    try:
        nmv = float(request.form["nmv"])
        tcp = float(request.form["tcp"])
        eia = float(request.form["eia"])
        thal = float(request.form["thal"])
        op = float(request.form["op"])
        mhra = float(request.form["mhra"])
        age = float(request.form["age"])
    except (KeyError, ValueError):
        flash("Invalid input values")
        return redirect(request.url)

    features = np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1)
    pred = heart_model.predict(features)[0]

    return render_template(
        "resulth.html",
        fn=firstname,
        ln=lastname,
        age=age,
        r=pred,
        gender=gender,
    )


# ------------------------- No Cache -------------------------


@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response


# ------------------------- Render / Gunicorn Entry -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
