import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """Create a TestClient where startup events (load_model) get triggered."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    # Checks that the health endpoint returns status OK
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_single_valid(client):
    # Tests single text classification with valid input
    payload = {"text": "Check my account balance"}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_classify_single_empty(client):
    # Ensures empty text payload returns a 400 error
    payload = {"text": ""}
    response = client.post("/api/classify", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Input text cannot be empty"


def test_classify_batch(client):
    # Tests batch classification endpoint for multiple inputs
    payload = {"texts": ["Check my balance", "Transfer money to friend"]}
    response = client.post("/api/classify/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for item in data:
        assert "text" in item
        assert "intent" in item
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0


def test_model_info_auth(client):
    # Checks model info endpoint with correct authentication
    response = client.get("/api/model/info", auth=("admin", "password"))
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "vectorizer_type" in data
    assert "num_classes" in data
    assert "classes" in data


def test_model_info_auth_fail(client):
    # Ensures model info endpoint fails with wrong authentication
    response = client.get("/api/model/info", auth=("wrong", "wrong"))
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]
