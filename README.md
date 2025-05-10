# TikTok Claim Classification Project

This project provides a machine learning model and API for classifying TikTok videos as either "claims" or "opinions". It uses natural language processing and engagement metrics to make predictions.

## Project Structure

```
Tiktok Claim Classification Project/
├── fast_api_app/                # FastAPI application
│   ├── main.py                  # FastAPI application definition
│   ├── preprocessing.py         # Data preprocessing logic
│   ├── schemas.py               # Pydantic models for API
│   └── models/                  # Directory for saved model files
├── notebooks/                   # Jupyter notebooks for analysis
├── data/                        # Data directory
│   └── tiktok_dataset.csv       # TikTok dataset
├── train_model.py               # Script to train and save model
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+ 
- pip (Python package manager)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model and save it for API use:

```bash
python train_model.py
```

This will:
1. Load and preprocess the TikTok dataset
2. Train an XGBoost classifier with hyperparameter tuning
3. Save both the model and text vectorizer to `fast_api_app/models/`

### Running the API

After training the model, start the FastAPI server:

```bash
uvicorn fast_api_app.main:app --reload
```

The API will be available at http://localhost:8000

### API Endpoints

- `GET /` - Root endpoint, returns API status
- `GET /health` - Health check endpoint
- `POST /predict` - Make a prediction for a video

### Example of using the API

To test the `/predict` endpoint with curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "video_id": 7017666017,
  "video_duration_sec": 59,
  "video_transcription_text": "someone shared with me that drone deliveries are predicted to become a common method of receiving packages in the future. studies show that companies are increasingly investing in drone technology.",
  "verified_status": "not verified",
  "author_ban_status": "active",
  "video_view_count": 343296.0,
  "video_like_count": 19425.0,
  "video_share_count": 241.0,
  "video_download_count": 1.0,
  "video_comment_count": 0.0
}'
```

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Model Information

The model uses:
- Text features from video transcription
- Engagement metrics (views, likes, shares, etc.)
- Classification is performed using an XGBoost classifier
- Hyperparameters are tuned using GridSearchCV with a focus on recall

## License

[MIT License](LICENSE)
