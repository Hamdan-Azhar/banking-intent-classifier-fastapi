from pydantic import BaseModel
from typing import List

# Pydantic models
class QueryRequest(BaseModel):
    text: str

class BatchQueryRequest(BaseModel):
    texts: List[str]

class QueryResponse(BaseModel):
    intent: str
    confidence: float

class BatchQueryResponseItem(BaseModel):
    text: str
    intent: str
    confidence: float

class ModelInfoResponse(BaseModel):
    model_name: str
    vectorizer_type: str
    num_classes: int
    classes: List[str]