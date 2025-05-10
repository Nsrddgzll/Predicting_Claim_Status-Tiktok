from pydantic import BaseModel, Field
from typing import Optional, Union, List


class VideoInput(BaseModel):
    """
    Schema for input data when making a prediction.
    Includes all necessary fields from the original dataset.
    """
    video_id: int = Field(..., description="Unique identifier for the video")
    video_duration_sec: int = Field(..., description="Duration of the video in seconds")
    video_transcription_text: str = Field(..., description="Transcription of the video content")
    verified_status: str = Field(..., description="Whether the user is verified or not")
    author_ban_status: str = Field(..., description="Ban status of the author: active, under review, or banned")
    video_view_count: float = Field(..., description="Number of views for the video")
    video_like_count: float = Field(..., description="Number of likes for the video")
    video_share_count: float = Field(..., description="Number of shares for the video")
    video_download_count: float = Field(..., description="Number of downloads for the video")
    video_comment_count: float = Field(..., description="Number of comments for the video")

    class Config:
        schema_extra = {
            "example": {
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
            }
        }


class PredictionOutput(BaseModel):
    """
    Schema for prediction output.
    """
    video_id: int = Field(..., description="The unique identifier of the video")
    claim_status: str = Field(..., description="The predicted label: 'claim' or 'opinion'")
    claim_probability: float = Field(..., description="Probability that the video contains a claim (0-1)")
    confidence: float = Field(..., description="Confidence in the prediction (0-1)")
    model_version: str = Field(..., description="Version of the model used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "video_id": 7017666017,
                "claim_status": "claim",
                "claim_probability": 0.92,
                "confidence": 0.92,
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """
    Schema for health check response.
    """
    status: str = Field(..., description="Status of the API")
    model_version: str = Field(..., description="Version of the model being used")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok", 
                "model_version": "1.0.0"
            }
        } 