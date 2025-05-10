#!/usr/bin/env python3
"""
Model training script for TikTok claim classification.

This script loads the TikTok dataset, preprocesses it, trains an XGBoost classifier 
with hyperparameter tuning, and saves the resulting model and text vectorizer.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Import preprocessing functions
from fast_api_app.preprocessing import load_and_clean_data, prepare_features_and_target

# Ensure the models directory exists
os.makedirs('fast_api_app/models', exist_ok=True)

def train_and_save_model(data_path='data/tiktok_dataset.csv'):
    """
    Train the model and save it for later use.
    
    Args:
        data_path: Path to the TikTok dataset CSV
    """
    print(f"Loading and cleaning data from {data_path}...")
    # Load and clean data
    data = load_and_clean_data(data_path)
    print(f"Data loaded and cleaned. Shape: {data.shape}")
    
    # Prepare features and target, fit text vectorizer
    print("Preparing features and target...")
    X, y, text_vectorizer = prepare_features_and_target(
        data, 
        fit_text_vectorizer=True
    )
    print(f"Features prepared. X shape: {X.shape}")
    
    # Split data into train, validation, and test sets (60/20/20)
    print("Splitting data into train, validation, and test sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )
    
    print(f"Train set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")
    
    # Train XGBoost model with GridSearchCV
    print("Training XGBoost model with GridSearchCV...")
    
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    # Initialize XGBoost classifier
    xgb = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scoring,
        refit='recall',  # Prioritize recall for final model selection
        cv=5,
        verbose=2
    )
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best recall: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    
    # Print validation metrics
    print("\nValidation Metrics:")
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Print classification report
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred))
    
    # Evaluate on test set
    y_test_pred = best_model.predict(X_test)
    
    # Print test metrics
    print("\nTest Metrics:")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Print classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    model_path = 'fast_api_app/models/tiktok_xgboost_model.joblib'
    vectorizer_path = 'fast_api_app/models/text_vectorizer.joblib'
    
    joblib.dump(best_model, model_path)
    joblib.dump(text_vectorizer, vectorizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":
    train_and_save_model() 