import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="Intent Classification",
    layout="centered",
)

st.title("💳 Banking Intent Classification")
st.caption("Classify customer banking queries using a TF-IDF–based Logistic Regression model.")

# Supported classes + examples
INTENT_EXAMPLES = {
    "card_payment_fee_charged": "I was charged an extra fee when I paid using my card.",
    "declined_cash_withdrawal": "My ATM withdrawal was declined even though I have enough balance.",
    "request_refund": "I would like a refund for an incorrect charge on my account.",
    "transaction_charged_twice": "My card was charged twice for the same transaction.",
    "transfer_not_received_by_recipient": "I sent money, but the recipient has not received it yet.",
    "wrong_amount_of_cash_received": "The ATM gave me less cash than I requested.",
}

# Sidebar (simple + informative)
with st.sidebar:
    st.header("📌 Supported Intents")
    for intent in INTENT_EXAMPLES.keys():
        st.markdown(f"- `{intent}`")

# Tabs
tab1, tab2 = st.tabs(["🔤 Single Text", "📄 Batch Classification"])

# Single Text Classification
with tab1:
    st.subheader("Single Text Classification")

    selected_intent = st.selectbox(
        "Select an intent category",
        options=list(INTENT_EXAMPLES.keys())
    )

    text = st.text_area(
        "Enter text or use the example below",
        value=INTENT_EXAMPLES[selected_intent],
        height=120
    )

    if st.button("🚀 Classify", key="single"):
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
                
                # Display predicted intent and confidence neatly in one line
                st.markdown(
                    f"""
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size:18px; font-weight:bold;">{data['intent']}</span>
                        <span style="font-size:16px;">Confidence: {data['confidence']:.4f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.progress(min(float(data["confidence"]), 1.0))
            else:
                st.error(response.json().get("detail", "Error occurred"))

# Batch Classification
with tab2:
    st.subheader("Batch Text Classification")
    st.caption("Enter one text per line")

    raw_texts = st.text_area(
        "Batch input",
        height=200,
        placeholder="Text 1\nText 2\nText 3"
    )

    if st.button("📊 Classify Batch", key="batch"):
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
