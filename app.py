"""
Flask web application for Potato Leaf Disease Classification.

Loads a pre-trained CNN model (model.keras) and serves a simple web
interface where a user can upload a photo of a potato leaf and receive
a predicted disease class along with a confidence score.
"""

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np


app = Flask(__name__, template_folder="templates")

# Ensure the upload/static directory exists before saving any files to it
os.makedirs('static', exist_ok=True)

# Load the trained model once at startup
model = tf.keras.models.load_model('model.keras')

# --- Configuration ---
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255
CHANNELS = 3
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CONFIDENCE_THRESHOLD = 0.80  # Below this, the image is treated as "not a potato leaf"


def allowed_file(filename):
    """Return True if the filename has one of the allowed image extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(img):
    """
    Run the loaded model on a single PIL image and return a
    (predicted_class, confidence_percentage) tuple.

    If the model's confidence is below CONFIDENCE_THRESHOLD, the image
    is reported as "not a potato leaf" instead of forcing a low-confidence
    guess.
    """
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    confidence_value = np.max(predictions[0])

    if confidence_value < CONFIDENCE_THRESHOLD:
        predicted_class = "This is not a potato leaf"
    else:
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = round(100 * confidence_value, 2)
    return predicted_class, confidence


@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle image upload (POST) and render the prediction result."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # Browser submits an empty file part if the user selects nothing
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(
                filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE)
            )
            predicted_class, confidence = predict(img)

            return render_template(
                'index.html',
                image_path=filepath,
                predicted_label=predicted_class,
                confidence=confidence,
            )

        return render_template('index.html', message='Unsupported file type')

    return render_template('index.html', message='Upload an image')


if __name__ == '__main__':
    # debug=True is for local development only — disable before deploying to production
    app.run(debug=True)
