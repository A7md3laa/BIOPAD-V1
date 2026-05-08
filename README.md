# BIOPAD-V1: Presentation Attack Detection System

**Author:** Ahmed Alaa  
**Institution:** University of Baghdad, College of Engineering (Electronics and Communication Dept.)  
**Performance:** 80.48% Global Accuracy | 0.92 AUC  

## Overview
BIOPAD-V1 is an early-interception biometric security checkpoint. It utilizes a Spatial Gabor filter bank and a custom cascading Deep Neural Network (DNN) to detect presentation attacks (forged/altered fingerprints) before they reach vulnerable traditional minutiae matchers.

## Project Structure
* `/backend`: FastAPI REST service and core inference logic (`extract_features`, `build_model`).
* `/frontend`: Interactive HTML/CSS diagnostic terminal.
* `/models`: Pre-trained Keras model weights.
* `/offline_tools`: Streamlit-based diagnostic prototype.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the FastAPI Server: `uvicorn backend.main:app --reload`
3. Access the Terminal: Open `http://localhost:8000` in your browser.
4. Run the Streamlit Prototype: `streamlit run offline_tools/app.py`
