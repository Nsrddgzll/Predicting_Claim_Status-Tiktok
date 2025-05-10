import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import schemas and preprocessing functions
from .schemas import VideoInput, PredictionOutput, HealthResponse
from .preprocessing import prepare_single_prediction

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
    Make a prediction for a video.
    
    Takes video data as input and returns whether it's a claim or opinion,
    along with the probability.
    """
    try:
        # Prepare input data for prediction
        input_data = video.dict()
        
        # Add dummy value for '#' column which will be dropped
        input_data["#"] = 0
        
        # Process input data
        X_pred = prepare_single_prediction(input_data, text_vectorizer)
        
        # Make prediction
        pred_proba = model.predict_proba(X_pred)[0]
        pred_class_idx = int(pred_proba[1] > 0.5)  # 1 if prob > 0.5 else 0
        
        # Map prediction to human-readable label (0=opinion, 1=claim)
        prediction = "claim" if pred_class_idx == 1 else "opinion"
        probability = pred_proba[pred_class_idx]
        
        return PredictionOutput(
            prediction=prediction,
            probability=float(probability)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 