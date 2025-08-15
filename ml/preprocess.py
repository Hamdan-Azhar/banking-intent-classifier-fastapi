import re

def clean_text(text: str):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Remove extra whitespaces
    text = re.sub("\s+", " ", text).strip()
    return text