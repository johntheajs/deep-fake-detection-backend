from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deep-fake-detection-frontend-johntheajs-projects.vercel.app",
        "https://deep-fake-detection-frontend-git-main-johntheajs-projects.vercel.app",
        "https://deep-fake-detection-backend.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the discriminator model
discriminator = tf.keras.models.load_model('Model/gan_discriminator_model.h5')

class ImageUrl(BaseModel):
    imageUrl: str

@app.post("/predict/")
async def predict(image_url: ImageUrl):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url.imageUrl)
        img = Image.open(io.BytesIO(response.content)).convert('L')  # Grayscale conversion

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    # Resize image
    img = img.resize((28, 28))

    # Convert to numpy array
    img_array = np.array(img)
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    label_array = np.array([1]).reshape(-1, 1)  # Dummy label

    # Make prediction
    prediction = discriminator([img_array, label_array], training=False)

    # Interpret the prediction
    result = "REAL" if prediction > 0 else "FAKE"
    return {"prediction": result}


# # If running locally, start Uvicorn server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
