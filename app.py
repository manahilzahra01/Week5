import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -----------------------------
# Page Title
# -----------------------------
st.title("Amazon Reviews Sentiment Analysis")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("DataSet (W5).csv")

# -----------------------------
# Load Model and Vectorizer
# -----------------------------
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# Text Cleaning Function
# -----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    if pd.isnull(text):
        return ""

    text = text.lower()

    text = re.sub(r'http\S+|www\S+', '', text)

    text = re.sub(r'\d+', '', text)

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = text.strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.selectbox(
    "Select a Page",
    ["Home", "Data Overview", "Sentiment Predictor"]
)

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":

    st.header("Welcome")

    st.write("""
This project performs Sentiment Analysis on Amazon Customer Reviews.

The machine learning model predicts whether a review is Positive or Negative.
""")

    st.subheader("Dataset")

    st.write("""
The dataset contains Amazon customer reviews along with sentiment labels.
""")

# -----------------------------
# Data Overview Page
# -----------------------------
elif page == "Data Overview":

    st.header("Data Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(5,4))

    df["feedback"].value_counts().plot(kind="bar", ax=ax)

    ax.set_xlabel("Feedback")
    ax.set_ylabel("Count")

    st.pyplot(fig)

    st.subheader("Word Clouds")

    # Clean reviews and convert to strings
    positive_reviews = " ".join(
        df[df["feedback"] == 1]["verified_reviews"]
        .fillna("")
        .apply(clean_text)
        .tolist()
    )

    negative_reviews = " ".join(
        df[df["feedback"] == 0]["verified_reviews"]
        .fillna("")
        .apply(clean_text)
        .tolist()
    )

    positive_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(positive_reviews)

    negative_wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(negative_reviews)

    fig, ax = plt.subplots(1, 2, figsize=(15,6))

    ax[0].imshow(positive_wc, interpolation="bilinear")
    ax[0].axis("off")
    ax[0].set_title("Positive Reviews")

    ax[1].imshow(negative_wc, interpolation="bilinear")
    ax[1].axis("off")
    ax[1].set_title("Negative Reviews")

    st.pyplot(fig)

# -----------------------------
# Sentiment Predictor
# -----------------------------
elif page == "Sentiment Predictor":

    st.header("Sentiment Predictor")

    review = st.text_area("Enter your review")

    if st.button("Predict"):

        cleaned_review = clean_text(review)

        review_vector = vectorizer.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]

        confidence = model.predict_proba(review_vector).max()

        if prediction == 1:
            st.success("😊 Positive Review")

        else:
            st.error("😞 Negative Review")

        st.write("Confidence Score:", round(confidence * 100, 2), "%")