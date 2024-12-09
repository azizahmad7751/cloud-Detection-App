# -*- coding: utf-8 -*-
"""
Created by

Author: Aziz Ahmad
"""

from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Load all the trained models
model_paths = {
    "CNN Base Model": 'model/modF6_base.h5',
    "Resnet50": 'model/modF6_resnet50.h5',
    "VGG16": 'model/modF6_vgg16.h5',
    "MobileNet_V2": 'model/modF6_mobilenetv2.h5',
}

models = {name: load_model(path) for name, path in model_paths.items()}

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, input_shape):
    """Load and preprocess the image for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_shape[1], input_shape[2]))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict_with_models(image_path):
    """Predict with all models and return their results."""
    predictions = {}
    for name, model in models.items():
        input_shape = model.input_shape
        processed_image = preprocess_image(image_path, input_shape)
        prediction = model.predict(processed_image)[0][0]
        result = "Cloud Detected" if prediction > 0.5 else "No Cloud Detected"
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        predictions[name] = {"result": result, "confidence": confidence}
    return predictions

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact page route
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle contact form submission
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        flash('Thank you for reaching out! We will get back to you shortly.')
        return redirect(url_for('contact'))
    return render_template('contact.html')    

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle the file upload and prediction."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                # Predict with all models
                predictions = predict_with_models(file_path)
                return render_template('results.html', predictions=predictions, filename=filename)
            except ValueError as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    """Display the uploaded image."""
    return redirect(url_for('static', filename='images/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
