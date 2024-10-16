from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2

# Initialize FastAPI app
app = FastAPI()

# Increase file size limit (set to 50 MB for example)
@app.middleware("http")
async def limit_file_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 50 * 1024 * 1024:  # 50 MB limit
            return JSONResponse(content={"detail": "File size too large"}, status_code=413)
    return await call_next(request)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained discriminator model when the app starts
discriminator = tf.keras.models.load_model('Model/gan_discriminator_model.h5')

# Utility function to preprocess image for prediction
def preprocess_image(image: Image.Image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)
    img_array = (img_array - 127.5) / 127.5  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=-1)  # Ensure correct channel dimension (1 channel)
    return img_array

# Define the /predict/image endpoint for image files
@app.post("/predict/image/")
async def predict_image(image: UploadFile = File(...)):
    # Ensure the uploaded file is an image
    try:
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes))  # Open image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img_array = preprocess_image(img)

    # Prepare label (assuming label=1 for real image)
    label = 1
    label_array = np.array([label]).reshape(-1, 1)  # Shape: (1, 1)

    # Make prediction using the discriminator model
    prediction = discriminator([img_array, label_array], training=False)

    # Interpret the prediction
    result = "REAL" if prediction > 0 else "FAKE"

    # Return the prediction result
    return {"prediction": result}

@app.post("/predict/video/")
async def predict_video(video: UploadFile = File(...)):
    if not video:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        video_bytes = await video.read()
        if len(video_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        video_stream = io.BytesIO(video_bytes)
        video_stream.seek(0)

        # Save video to disk temporarily to read with OpenCV
        with open('temp_video.mp4', 'wb') as temp_video_file:
            temp_video_file.write(video_stream.read())
        
        # Open the video using OpenCV
        cap = cv2.VideoCapture('temp_video.mp4')
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to open video file")
    except Exception as e:
        raise HTTPException(status_code=(400), detail=f"Invalid video file: {str(e)}")

    # Rest of the frame processing and prediction logic...

    frame_predictions = []
    success, frame = cap.read()

    # Process each frame in the video
    while success:
        # Convert the frame to PIL Image for prediction
        frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_array = preprocess_image(frame_image)

        # Prepare label (assuming label=1 for real image)
        label = 1
        label_array = np.array([label]).reshape(-1, 1)

        # Make prediction using the discriminator model
        frame_prediction = discriminator([frame_array, label_array], training=False)
        frame_predictions.append(frame_prediction)

        # Read next frame
        success, frame = cap.read()

    cap.release()

    # Consolidate predictions for all frames
    real_count = sum(1 for pred in frame_predictions if pred > 0)
    fake_count = len(frame_predictions) - real_count

    # If more than 50% frames are predicted REAL, classify video as REAL, otherwise FAKE
    final_result = "REAL" if real_count > fake_count else "FAKE"

    # Return the final result for the video
    return {"prediction": final_result, "real_frames": real_count, "fake_frames": fake_count}


# If running locally, start Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
