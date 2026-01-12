import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="Intent Classification", layout="centered")
st.title("Intent Classification")

tab1, tab2 = st.tabs(["Single Text", "Batch Classification"])

# Single Text Classification Tab
with tab1:
    st.subheader("Classify Single Text")

    text = st.text_area("Enter text", height=120)

    if st.button("Classify", key="single"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Classifying..."):
                response = requests.post(
                    f"{API_BASE_URL}/classify",
                    json={"text": text}
                )

            if response.status_code == 200:
                data = response.json()
                st.success("Prediction successful")
                st.markdown(f"**Intent:** `{data['intent']}`")
                st.markdown(f"**Confidence:** `{data['confidence']:.4f}`")
            else:
                st.error(response.json().get("detail", "Error occurred"))

# Batch Classification Tab
with tab2:
    st.subheader("Batch Text Classification")

    raw_texts = st.text_area(
        "Enter one text per line",
        height=200,
        placeholder="Text 1\nText 2\nText 3"
    )

    if st.button("Classify Batch", key="batch"):
        texts = [t.strip() for t in raw_texts.split("\n") if t.strip()]

        if not texts:
            st.warning("Please enter at least one non-empty line.")
        else:
            with st.spinner("Classifying batch..."):
                response = requests.post(
                    f"{API_BASE_URL}/classify/batch",
                    json={"texts": texts}
                )

            if response.status_code == 200:
                results = response.json()
                df = pd.DataFrame(results)

                st.success(f"Classified {len(df)} texts")
                st.dataframe(df, use_container_width=True)
            else:
                st.error(response.json().get("detail", "Error occurred"))
