import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from typing import Tuple, Optional, Union, Dict, Any


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """
    Load TikTok data from CSV file and clean by removing rows with missing values.
    
    Args:
        csv_path: Path to the TikTok dataset CSV file
        
    Returns:
        Cleaned pandas DataFrame with no missing values
    """
    # Load dataset
    data = pd.read_csv(csv_path)
    
    # Drop rows with any missing values
    data = data.dropna(axis=0)
    
    return data


def prepare_features_and_target(
    df: pd.DataFrame, 
    text_vectorizer: Optional[CountVectorizer] = None,
    fit_text_vectorizer: bool = False
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[CountVectorizer]]:
    """
    Prepare features and target from the cleaned TikTok dataframe.
    
    Args:
        df: Cleaned DataFrame with no missing values
        text_vectorizer: Pre-fitted CountVectorizer (optional)
        fit_text_vectorizer: Whether to fit a new text vectorizer
        
    Returns:
        X_final: Processed features DataFrame
        y_final: Target Series (None if not available in input)
        text_vectorizer: The CountVectorizer (fitted if fit_text_vectorizer=True)
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Create text_length feature
    data['text_length'] = data['video_transcription_text'].str.len()
    
    # Check if claim_status is in the DataFrame (it won't be for prediction)
    has_target = 'claim_status' in data.columns
    
    # Prepare target if available
    y_final = None
    if has_target:
        # Encode target variable (0=opinion, 1=claim)
        y_final = data['claim_status'].replace({'opinion': 0, 'claim': 1})
    
    # Drop unnecessary columns for features
    X = data.drop(['#', 'video_id'] + (['claim_status'] if has_target else []), axis=1, errors='ignore')
    
    # Handle text vectorization
    if fit_text_vectorizer:
        # Initialize and fit a new vectorizer
        text_vectorizer = CountVectorizer(
            ngram_range=(2, 3),
            max_features=15,
            stop_words='english'
        )
        count_data = text_vectorizer.fit_transform(X['video_transcription_text']).toarray()
    elif text_vectorizer is not None:
        # Use provided vectorizer
        count_data = text_vectorizer.transform(X['video_transcription_text']).toarray()
    else:
        raise ValueError("Either fit_text_vectorizer must be True or a pre-fitted text_vectorizer must be provided")
    
    # Convert token counts to DataFrame
    count_df = pd.DataFrame(
        data=count_data,
        columns=text_vectorizer.get_feature_names_out()
    )
    
    # Concatenate features with tokens, dropping the original text column
    X_processed = pd.concat(
        [X.drop(columns=['video_transcription_text']).reset_index(drop=True), 
         count_df], 
        axis=1
    )
    
    # One-hot encode categorical variables
    X_final = pd.get_dummies(
        X_processed,
        columns=['verified_status', 'author_ban_status'],
        drop_first=True
    )
    
    return X_final, y_final, text_vectorizer


def prepare_single_prediction(
    input_data: Dict[str, Any], 
    text_vectorizer: CountVectorizer
) -> pd.DataFrame:
    """
    Prepares a single input for prediction with all necessary feature engineering.
    
    Args:
        input_data: Dictionary containing input data from the API request
        text_vectorizer: Pre-trained CountVectorizer for text features
    
    Returns:
        DataFrame with all features needed for model prediction
    """
    # Create a pandas Series from input data
    input_series = pd.Series(input_data)
    
    # Extract numerical features
    numerical_features = [
        'video_duration_sec', 
        'video_view_count', 
        'video_like_count',
        'video_share_count', 
        'video_download_count', 
        'video_comment_count'
    ]
    
    # Create a DataFrame with numerical features
    X = pd.DataFrame([input_series[numerical_features].values], columns=numerical_features)
    
    # Add text length feature
    X['text_length'] = len(input_data['video_transcription_text'])
    
    # Add text features using vectorizer
    if input_data['video_transcription_text']:
        text_features = text_vectorizer.transform([input_data['video_transcription_text']])
        text_feature_names = text_vectorizer.get_feature_names_out()
        text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)
        X = pd.concat([X, text_df], axis=1)
    else:
        # If empty text, add zero columns for text features
        text_feature_names = text_vectorizer.get_feature_names_out()
        zeros = np.zeros((1, len(text_feature_names)))
        text_df = pd.DataFrame(zeros, columns=text_feature_names)
        X = pd.concat([X, text_df], axis=1)
    
    # Handle categorical features using one-hot encoding
    # Add verified_status feature
    X['verified_status_verified'] = 1 if input_data['verified_status'] == 'verified' else 0
    
    # Add author_ban_status features
    X['author_ban_status_banned'] = 1 if input_data['author_ban_status'] == 'banned' else 0
    X['author_ban_status_under review'] = 1 if input_data['author_ban_status'] == 'under review' else 0
    
    return X 