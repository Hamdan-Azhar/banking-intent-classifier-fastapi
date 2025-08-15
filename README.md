# Intent Classification API (TF-IDF + Logistic Regression)

A lightweight FastAPI service that classifies user intents in **online banking queries** using a trained **TF-IDF + Logistic Regression** model. Access the interactive Swagger UI at the `/docs` endpoint.  
Detailed Model development, experimentation, performance reports and visualization are documented in `ml/training.ipynb`. 

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. To run the FastAPI server
uvicorn api.main:app --reload

# 5. To run tests
pytest

```

## 🧪 Jupyter notebook 

It is recommended to run the notebook in Google Colab.
Alternatively, create a new virtual environment in VS Code or Jupyter before running the notebook. 





