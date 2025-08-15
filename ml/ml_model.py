import pickle, os

def load_model():

     # Get directory of this file
    base_dir = os.path.dirname(os.path.abspath(__file__))

    vectorizer_path = os.path.join(base_dir, "models/tfidf_vectorizer.pkl")
    model_path = os.path.join(base_dir, "models/logistic_regression_model.pkl")

    # Load the model and vectorizer and return it    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model, vectorizer
