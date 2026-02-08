# ML Text Classification API

## Project Overview

This project implements a machine learning-based text classification API using FastAPI.  
The API classifies text messages as either **Spam** or **Not Spam** using a trained Logistic Regression classifier with TF-IDF vectorization.

---

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs in Python  
- **scikit-learn**: Machine learning library for model training and TF-IDF vectorization  
- **Pydantic**: Data validation and serialization for request/response handling  
- **Uvicorn**: ASGI server for running FastAPI application  
- **pandas & numpy**: Data manipulation and numerical computing  
- **pickle**: Serialization for saving and loading trained models  

---

## Setup Instructions

1. **Clone the repository** (if from GitHub):
```bash
git clone https://github.com/Manal5664/fastapi-spam-detection
cd Project_Fast_Api
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3.**Run the training notebook (optional, if you want to retrain the model):**
```bash
jupyter notebook train_model.ipynb

```
4.**Start the FastAPI server:**
```bash

python -m uvicorn main:app --reload
```

## API Endpoints

---

### 1. Health Check

- **Endpoint:** `GET /health`  
- **Description:** Returns the status of the API.  
- **Response Example:**

```json
{
  "status": "API is running"
}

```

### 2. Root Endpoint

- **Endpoint:** `GET /`  
- **Description:** Returns a welcome message and available endpoints.  
- **Response Example:**

```json
{
  "message": "Welcome to ML Text Classification API",
  "endpoints": {
    "health": "/health (GET)",
    "predict": "/predict (POST)"
  }
}
```

### 3. Prediction

- **Endpoint:** `POST /predict`  
- **Description:** Classifies input text as **Spam** or **Not Spam**.  

**Request Body Example:**

```json
{
  "text": "You have won a free iPhone! Click here NOW!!!"
}
```



