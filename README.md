# Banking Intent Classification System 🏦🤖

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-📡-green)](https://fastapi.tiangolo.com/)

A **Banking Intent Classification system** built using classical NLP techniques and deployed as a **FastAPI REST service**. The project demonstrates the full lifecycle of an NLP model: data loading, preprocessing, feature engineering, training, evaluation, and deployment.

---

## 🔄 End-to-End System Flow

```text
User Query
   ↓
FastAPI REST Endpoint (/predict)
   ↓
Text Preprocessing
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression Classifier
   ↓
Predicted Intent + Confidence Score
```

After training and evaluation of the preprocesser and vectorizer, their weights are stored and then called with the API layer, making it usable in real banking systems such as chatbots and customer-support automation.

## 🎯 Problem Statement

In digital banking systems, user inputs are typically short, ambiguous, and noisy (e.g., *"check my balance"*, *"transfer money"*). Accurately identifying the **intent** behind such queries is essential for automation, chatbots, and intelligent customer support.

This project formulates intent recognition as a **multi‑class text classification problem**, mapping raw text queries to predefined banking intents.

---

## 🧠 Machine Learning Pipeline

All experimentation and model development are documented in:

```
ml/training.ipynb
```

### 1️⃣ Data Loading & Exploration

* Dataset is loaded using **Pandas**
* Initial inspection includes:

  * Class distribution
  * Sample utterances per intent
  * Missing / noisy text handling

📌 *Suggested visual*:
`Class distribution bar chart` — helps reviewers immediately see dataset balance.

---

### 2️⃣ Text Preprocessing

The raw banking queries undergo standard NLP preprocessing:

* Lower‑casing
* Removal of punctuation and special characters
* Token normalization

This step reduces vocabulary noise while preserving semantic intent.

📌 *Suggested visual*:
`Before vs after preprocessing examples` (small table or screenshot).

---

### 3️⃣ Feature Engineering — TF‑IDF

Text is transformed into numerical vectors using **TF‑IDF (Term Frequency–Inverse Document Frequency)**:

* Captures importance of words relative to intent classes
* Prevents dominance of common but uninformative tokens
* Produces a sparse, high‑dimensional representation suitable for linear models

This choice reflects a deliberate trade‑off:

> interpretability and robustness over unnecessary model complexity.

📌 *Suggested visuals*:

* WordCloud per intent class
* Top‑N TF‑IDF features per class (table or bar plot)

---

### 4️⃣ Model Training — Logistic Regression

A **Logistic Regression classifier** is trained on the TF‑IDF vectors:

* Strong baseline for text classification
* Fast to train, easy to debug
* Coefficients directly reflect feature importance

The notebook documents:

* Train / test split
* Model fitting
* Hyperparameter defaults (kept intentionally simple)

📌 *Suggested visual*:
`Model training workflow diagram`

---

### 5️⃣ Model Evaluation

Performance is evaluated using standard classification metrics:

* Accuracy
* Precision, Recall, F1‑Score
* Confusion Matrix

This provides both **quantitative performance** and **qualitative error analysis**.

📌 *Highly recommended visuals*:

* Confusion Matrix heatmap
* Classification report screenshot
* Misclassified examples table

These visuals strongly signal analytical maturity to admissions committees.

---

## 🧩 API Architecture

Endpoints are exposed under the `api/` prefix. These endpoints operationalize the trained NLP model and provide both inference and system-level metadata, following clean API design principles.

### 📌 Available Endpoints (`api/`)

#### `GET /api/health`
Lightweight health-check endpoint.

- Confirms that the API service is running.
- Useful for deployment monitoring and orchestration systems.

---

#### `POST /api/classify`
Single-text intent classification endpoint.

- Accepts a single banking-related text query.
- Applies the same preprocessing and TF-IDF vectorization used during training.
- Uses the trained Logistic Regression model to infer intent.
- Returns the predicted intent label along with a confidence score.

This endpoint represents the primary inference path for real-time applications such as chatbots.

---

#### `POST /api/classify/batch`
Batch intent classification endpoint.

- Accepts multiple text queries in a single request.
- Processes each query independently through the same NLP pipeline.
- Returns intent predictions and confidence scores for each input.

Designed for offline analysis, bulk evaluation, or integration with data pipelines.

---

#### `GET /api/model/info` *(Protected)*
Model metadata and inspection endpoint.

- Requires HTTP Basic authentication.
- Exposes model and vectorizer details, including:
  - model type,
  - vectorizer type,
  - number of intent classes,
  - list of supported intent labels.

This endpoint improves transparency and supports debugging, auditing, and system introspection without exposing model internals.

---

### 🔐 Security Note

Sensitive endpoints are protected using **HTTP Basic Authentication**, demonstrating awareness of access control even in lightweight ML services.

---

## 🛠️ Running the Project

```bash
git clone https://github.com/Hamdan-Azhar/banking-intent-classifier-fastapi.git
cd banking-intent-classifier-fastapi
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Explore the API at:

```
http://localhost:8000/docs
```---

