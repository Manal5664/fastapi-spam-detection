# ML Text Classification API

## Project Overview

This project implements a machine learning-based text classification API using FastAPI. The API classifies text messages as either **Spam** or **Not Spam** using a trained Logistic Regression classifier with TF-IDF vectorization.
 
## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs with Python
- **scikit-learn**: Machine learning library for model training and TF-IDF vectorization
- **Pydantic**: Data validation and serialization for request/response handling
- **uvicorn**: ASGI server for running the FastAPI application
- **pandas & numpy**: Data manipulation and numerical computing
- **pickle**: Serialization for saving and loading trained models

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txtjupyter notebook train_model.ipynbuvicorn main:app --reload

Write a README.md for ML Text Classification API project.
API Endpoints
1. Health Check Endpoint
GET /health

Returns the status of the API.

Response:{
  "status": "API is running"
}
2. Root Endpoint
GET /

Returns welcome message and available endpoints.{
  "message": "Welcome to ML Text Classification API",
  "endpoints": {
    "health": "/health (GET)",
    "predict": "/predict (POST)"
  }
}

3. Prediction Endpoint
POST /predict

Classifies input text as Spam or Not Spam.

Request Body:{
  "text": "You have won a free iPhone! Click here NOW!!!"
}{
  "prediction": "Spam",
  "confidence": 0.95
}