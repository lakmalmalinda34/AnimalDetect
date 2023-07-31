import os
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

app = Flask(__name__)


# Load the pre-trained CNN model
model = load_model('Model')
# model._make_predict_function()  # Necessary for multi-threaded applications

# Function to perform image classification
def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_preprocessed = preprocess_input(img)

    result = model.predict(img_preprocessed)
    classes = ['Elephant', 'NotElephant']
    predicted_class = classes[np.argmax(result)]

    return predicted_class

# API endpoint to handle image classification
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Create 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file temporarily
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Perform image classification
    prediction = classify_image(file_path)

    # Remove the temporary uploaded file
    os.remove(file_path)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
