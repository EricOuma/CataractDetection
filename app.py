import os
from tensorflow.keras import models
import cv2

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

from flask import Flask, flash, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

modelFileName = 'static/models/cataract_classifier_v2.h5'
model = models.load_model(modelFileName)

img_height = 150
img_width = 150

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('diagnose', image_path=filename))
    return render_template('upload.html')


@app.route('/diagnose')
def diagnose():
    image_path = request.args.get('image_path')
    img = cv2.imread('uploads/'+image_path)
    resized_image = cv2.resize(img, (img_height, img_width))
    imgfeatures = resized_image.reshape(1, 150, 150, 3)
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255

    class_probabilities = model.predict(imgfeatures)
    # Find the class predictions with the highest predicted probability
    result = int([0 if x < 0.5 else 1 for x in class_probabilities][0])
    print(f'Result: {result}{type(result)}')
    return redirect(url_for('feedback', result=result))
    # return render_template('diagnose.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html', result=int(request.args.get('result')))