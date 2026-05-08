from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Import our separated core logic
from .core.extract_features import extract_features_from_array

app = FastAPI()

# Establish absolute paths to reach other folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend/static")
MODEL_PATH = os.path.join(BASE_DIR, "../models/Fingerprint_Recognition_Model_24k.keras")

# Mount the static folder to serve the HTML UI
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Load model globally
MODEL = load_model(MODEL_PATH)

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/scan")
async def scan_fingerprint(file: UploadFile = File(...)):
    # Read Image from request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Process through core logic
    input_vector = extract_features_from_array(img)
    
    # Predict
    prob = float(MODEL.predict(input_vector)[0][0])
    
    # Security Threshold (0.26)
    status = "ALTERED" if prob > 0.26 else "REAL"
    
    return {"probability": prob, "status": status}
