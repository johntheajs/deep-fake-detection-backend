from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained discriminator model when the app starts
discriminator = tf.keras.models.load_model('Model/gan_discriminator_model.h5')

# Define the /predict endpoint
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    # Ensure the uploaded file is an image
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Resize the image to (28, 28)
    img = img.resize((28, 28))

    # Convert image to numpy array using np.array
    img_array = np.array(img)

    # Normalize the image
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=-1)  # Ensure correct channel dimension

    label = 1
    # Prepare the label
    label_array = np.array([label]).reshape(-1, 1)  # Shape: (1, 1)

    # Make prediction using the discriminator model
    prediction = discriminator([img_array, label_array], training=False)

    # Interpret the prediction
    result = "REAL" if prediction > 0 else "FAKE"

    # Return the prediction result
    return {"prediction": result}

# If running locally, start Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
