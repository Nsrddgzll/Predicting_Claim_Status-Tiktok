import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import schemas and preprocessing functions
from .schemas import VideoInput, PredictionOutput, HealthResponse
from .preprocessing import prepare_single_prediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define application
app = FastAPI(
    title="TikTok Claim Classification API",
    description="API for classifying TikTok videos as claims or opinions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Model version
MODEL_VERSION = "1.0.0"

# Load model and vectorizer at startup
@app.on_event("startup")
async def load_model():
    global model, text_vectorizer
    
    model_path = os.path.join("fast_api_app", "models", "tiktok_xgboost_model.joblib")
    vectorizer_path = os.path.join("fast_api_app", "models", "text_vectorizer.joblib")
    
    # Check if model and vectorizer files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise RuntimeError(
            "Model or vectorizer files not found. Please run train_model.py first."
        )
    
    # Load model and vectorizer
    model = joblib.load(model_path)
    text_vectorizer = joblib.load(vectorizer_path)


@app.get("/", response_model=HealthResponse)
async def root():
    """
    Root endpoint, provides basic API information.
    """
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify the API is running properly.
    """
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionOutput)
async def predict(video: VideoInput):
    """
    Predict whether a TikTok video contains a claim or opinion
    """
    try:
        # Prepare input data
        input_data = video.dict()
        
        # Log the incoming request
        logger.info(f"Received prediction request for video_id: {input_data['video_id']}")
        
        # Prepare features for prediction
        X_pred = prepare_single_prediction(input_data, text_vectorizer)
        
        # Log the feature names to help with debugging
        logger.info(f"Feature names after preprocessing: {list(X_pred.columns)}")
        
        # Make prediction
        prediction_proba = model.predict_proba(X_pred)[0]
        prediction = model.predict(X_pred)[0]
        
        # Map numerical prediction to label
        claim_status = "claim" if prediction == 1 else "opinion"
        
        # Calculate confidence
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        return {
            "video_id": input_data["video_id"],
            "claim_status": claim_status,
            "claim_probability": float(prediction_proba[1]),
            "confidence": float(confidence),
            "model_version": MODEL_VERSION
        }
    except Exception as e:
        # Log the error
        logger.error(f"Prediction error: {str(e)}")
        
        # If it's a feature mismatch error, provide more details
        if "feature_names mismatch" in str(e):
            model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
            input_features = list(X_pred.columns) if 'X_pred' in locals() else []
            
            missing_features = [f for f in model_features if f not in input_features]
            extra_features = [f for f in input_features if f not in model_features]
            
            error_detail = f"Feature mismatch error: Missing {missing_features}, Extra {extra_features}"
            logger.error(error_detail)
            
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 