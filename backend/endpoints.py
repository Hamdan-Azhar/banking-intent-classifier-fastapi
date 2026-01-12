from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from .models import QueryRequest, QueryResponse, BatchQueryRequest, BatchQueryResponseItem, ModelInfoResponse
from typing import List
import numpy as np
from ml.preprocess import clean_text
import secrets

# Initialize the router that will hold all API endpoints
router = APIRouter()

# Set up HTTP basic authentication for protected routes
security = HTTPBasic()

# Authentication setup (use it when calling model/info api)
USERNAME = "admin"
PASSWORD = "password"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


# Classification for single text
@router.post("/classify", response_model=QueryResponse)
def classify_single(query: QueryRequest, request:Request):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    processed_text = clean_text(query.text)

    vectorizer = request.app.state.vectorizer
    model = request.app.state.model

    X_vec = vectorizer.transform([processed_text])
    
    # Find max confidence from model predictions
    pred_probs = model.predict_proba(X_vec)[0]
    pred_idx = np.argmax(pred_probs)
    confidence = float(pred_probs[pred_idx])
    
    # Return class predicted with maximum confidence
    intent = model.classes_[pred_idx]
    return {"intent": intent, "confidence": confidence}

# CLassification for multiple inputs in a batch
@router.post("/classify/batch", response_model=List[BatchQueryResponseItem])
def classify_batch(query: BatchQueryRequest, request:Request):
    results = []
    for text in query.texts:

        # Preprocess text
        processed_text = clean_text(text)

        vectorizer = request.app.state.vectorizer
        model = request.app.state.model

        X_vec = vectorizer.transform([processed_text])
        
        # Find max confidence from model predictions
        pred_probs = model.predict_proba(X_vec)[0]
        pred_idx = np.argmax(pred_probs)
        confidence = float(pred_probs[pred_idx])
        
        # Return class predicted with maximum confidence
        intent = model.classes_[pred_idx]
        results.append({"text": text, "intent": intent, "confidence": confidence})
    return results

# Health check
@router.get("/health")
def health_check():
    return {"status": "ok"}

# Model info (requires authentication)
@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(request: Request, auth: bool = Depends(authenticate)):

    # Retrieve loaded vectorizer and model from FastAPI app state
    vectorizer = request.app.state.vectorizer
    model = request.app.state.model
     
    return ModelInfoResponse(
        model_name=str(type(model).__name__),
        vectorizer_type=str(type(vectorizer).__name__),
        num_classes=len(model.classes_),
        classes=list(model.classes_)
    )
