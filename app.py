from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from tensorflow import keras
import h5py

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = (r'C:\Users\Ahmed\Desktop\Strawberry-segmentation\models')

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function() 
print('Model loaded. Check http://127.0.0.1:5000/')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input



def model_predict(image_path):
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image =  keras.preprocessing.image.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

  



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)
        if (preds<0):
               result="It's a strawberry!"
        else:
              result="That's not a strawberry!"
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

