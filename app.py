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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Load the trained model
model = load_model('model/modF6.h5')

def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Placeholder function for loading test data
def load_test_data():
    # Replace with actual code to load test images and labels
    test_images = []  # List of file paths for test images
    test_labels = []  # List of labels (0 for No Cloud, 1 for Cloud)
    return test_images, test_labels

def evaluate_model_performance(model):
    test_images, test_labels = load_test_data()
    predictions = []

    # Process each test image for predictions
    for image_path in test_images:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        desired_width, desired_height = model.input_shape[1:3]
        image = cv2.resize(image, (desired_width, desired_height))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict and append the result
        pred = model.predict(image)[0][0]
        predictions.append(1 if pred > 0.5 else 0)  # 1 for Cloud, 0 for No Cloud

    # Calculate performance metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    return accuracy, precision, recall, f1

def predict_cloud(image_path):
    """Predict whether an image contains clouds using the CNN model."""
    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    desired_width, desired_height = model.input_shape[1:3]
    image = cv2.resize(image, (desired_width, desired_height))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict cloud presence
    prediction = model.predict(image)[0][0]
    result = "Cloud Detected" if prediction > 0.5 else "No Cloud Detected"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    # Calculate model performance metrics
    accuracy, precision, recall, f1 = evaluate_model_performance(model)

    # Return the result along with performance metrics
    return result, confidence, accuracy, precision, recall, f1

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
        # Check if file is in the request
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
                # Make prediction and get performance metrics
                result, confidence, accuracy, precision, recall, f1 = predict_cloud(file_path)
                # Render result page with prediction and metrics
                return render_template('results.html', result=result, confidence=confidence, filename=filename)
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
