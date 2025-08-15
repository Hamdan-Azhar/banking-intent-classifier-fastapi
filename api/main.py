from fastapi import FastAPI
from api.endpoints import router as api_router
from ml.ml_model import load_model

app = FastAPI(title="Intent Classification API", version="1.0")


@app.on_event("startup")
def startup_event():
    model, vectorizer = load_model()
    app.state.model = model
    app.state.vectorizer = vectorizer
    print("Vectorizer and model loaded successfully!")

# Include endpoints
app.include_router(api_router, prefix='/api')
