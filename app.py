from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
from flask_cors import CORS



app = Flask(__name__)
CORS(app)


# Load the trained discriminator model when the app starts
discriminator = tf.keras.models.load_model('D:/Projects/CDP-BACKEND/Model/gan_discriminator_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    if 'label' not in request.form:
        return jsonify({'error': 'No label provided'}), 400

    # Get the image and label from the request
    img_file = request.files['image']
    label = request.form['label']

    # Convert label to integer
    try:
        label = int(label)
    except ValueError:
        return jsonify({'error': 'Label must be an integer (0 or 1)'}), 400

    # Read the image in bytes and open it with PIL
    try:
        img_bytes = img_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale
    except Exception as e:
        return jsonify({'error': 'Invalid image file'}), 400

    # Resize the image to match the input shape expected by the model
    img = img.resize((28, 28))

    # Convert the image to array
    img_array = image.img_to_array(img)

    # Normalize the image
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28, 1)

    # Prepare the label
    label_array = np.array([label]).reshape(-1, 1)  # Shape: (1, 1)

    # Make prediction
    prediction = discriminator([img_array, label_array], training=False)

    # Interpret the prediction
    if prediction > 0:
        result = 'REAL'
    else:
        result = 'FAKE'

    # Return the result
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
