from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="ML Text Classification API",
    description="API for spam vs ham text classification",
    version="1.0.0"
)

# Load trained model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Pydantic model for request
class TextInput(BaseModel):
    text: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: TextInput):
    """
    Predict if text is spam or not spam.
    
    Args:
        input_data: TextInput object containing 'text' field
    
    Returns:
        JSON with prediction ('Spam' or 'Not Spam') and confidence score
    """
    try:
        # Transform text using the vectorizer
        text_vectorized = vectorizer.transform([input_data.text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        
        # Get prediction probability
        confidence = model.predict_proba(text_vectorized)[0]
        
        # Map prediction to label
        label = "Spam" if prediction == 1 else "Not Spam"
        
        # Get confidence for the predicted class
        confidence_score = float(max(confidence))
        
        return {
            "prediction": label,
            "confidence": confidence_score
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "prediction": None,
            "confidence": None
        }

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to ML Text Classification API",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    }

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)