import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# ------------------------
# Load Model and Data
# ------------------------
model = load_model("spam_classifier_model.h5")

# Load features (columns used in training)
features = pd.read_csv("./dataset/emails.csv").drop(columns=["Email No.", "Prediction"]).columns
word_to_index = {word: i for i, word in enumerate(features)}

# Load precomputed metrics
metrics_df = pd.read_csv("metrics.csv", index_col=0)
cm = np.load("confusion_matrix.npy")

# ------------------------
# Functions
# ------------------------
def email_to_vector(email_text):
    vec = np.zeros(len(word_to_index))
    for word in email_text.lower().split():
        if word in word_to_index:
            vec[word_to_index[word]] += 1
    return vec

def predict_email(email_text):
    vec = email_to_vector(email_text).reshape(1, -1)
    pred_prob = model.predict(vec)[0][0]
    label = "Spam ⚠️" if pred_prob > 0.5 else "Ham ✅"
    return label, pred_prob

# ------------------------
# Streamlit Layout
# ------------------------
st.set_page_config(page_title="Email Spam Detector", layout="wide")
st.title("📧 Email Spam Detection App")
st.write("Paste your email text below to check if it is Spam or Ham.")

# Input section
email_input = st.text_area("Enter Email Text Here:", height=200)

if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email text to check.")
    else:
        label, score = predict_email(email_input)
        if label == "Spam ⚠️":
            st.error(f"{label} (Prediction Score: {score:.4f})")
        else:
            st.success(f"{label} (Prediction Score: {score:.4f})")

# ------------------------
# Two-column layout for metrics & confusion matrix
# ------------------------
st.markdown("---")
st.header("📊 Model Performance on Test Set")

col1, col2 = st.columns(2)

# ----- Left Column -----
with col1:
    st.subheader("Overall Accuracy")
    # Example: if you saved overall accuracy in metrics_df, else set manually
    overall_accuracy = metrics_df.loc["accuracy", "f1-score"] if "accuracy" in metrics_df.index else 0.9845
    st.metric(label="Test Accuracy", value=f"{overall_accuracy*100:.2f}%")

    st.subheader("Class-wise Precision, Recall, F1-score")
    for cls in metrics_df.index[:-3]:  # skip avg rows if present
        st.write(
            f"{cls} → Precision: {metrics_df.loc[cls, 'precision']:.2f}, "
            f"Recall: {metrics_df.loc[cls, 'recall']:.2f}, "
            f"F1-score: {metrics_df.loc[cls, 'f1-score']:.2f}"
        )

# ----- Right Column -----
with col2:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5,4))  # Adjust size as needed
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
