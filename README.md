# Amazon Reviews Sentiment Analysis using Natural Language Processing (NLP)

## Project Description

This project is developed as part of the Week 5 NLP Assignment. The main objective of this project is to analyze Amazon customer reviews and classify them into **Positive** or **Negative** sentiments using Machine Learning and Natural Language Processing (NLP).

The project is divided into two parts:

1. **Part 1:** Data preprocessing, feature extraction, model training, evaluation, and model saving using Google Colab.
2. **Part 2:** Development of an interactive Streamlit web application that allows users to predict the sentiment of new Amazon reviews using the saved model.


# Project Objectives

The objectives of this project are:

- Understand the process of Natural Language Processing.
- Clean and preprocess text data.
- Convert textual data into numerical features using TF-IDF Vectorization.
- Train and evaluate Machine Learning models.
- Save the best-performing model for future predictions.
- Build a user-friendly Streamlit application.
- Predict customer review sentiment with confidence scores.


# Dataset Information

**Dataset Name**

DataSet (W5).csv

The dataset contains Amazon customer reviews and sentiment labels.

## Main Columns

| Column Name | Description |
|-------------|-------------|
| verified_reviews | Customer review text |
| feedback | Sentiment label (1 = Positive, 0 = Negative) |


# Tools and Technologies Used

- Python
- Google Colab
- Visual Studio Code (VS Code)
- Streamlit
- Pandas
- NumPy
- Matplotlib
- WordCloud
- NLTK
- Scikit-learn
- Joblib


# Part 1 – NLP Model Development (Google Colab)

## Step 1: Import Libraries

The following Python libraries were imported:

- Pandas
- NumPy
- Matplotlib
- NLTK
- Scikit-learn
- Joblib
- Re
- String

These libraries were used for data preprocessing, visualization, machine learning, and model saving.


## Step 2: Load Dataset

The dataset was loaded using Pandas.

```python
df = pd.read_csv("DataSet (W5).csv")
```

## Step 3: Data Exploration

The dataset was explored by performing the following operations:

- Display first five rows
- Display last five rows
- Check dataset dimensions
- Check column names
- Check missing values
- Check duplicate values
- Display dataset information
- Display descriptive statistics

This helped in understanding the structure and quality of the dataset.


## Step 4: Data Cleaning

The review text was cleaned before model training.

The following preprocessing techniques were applied:

- Convert all text to lowercase
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove extra white spaces
- Remove stopwords
- Apply Lemmatization

This preprocessing improves the quality of the text data.


## Step 5: Feature Engineering

The cleaned text was converted into numerical vectors using:

**TF-IDF Vectorizer**

TF-IDF helps Machine Learning algorithms understand textual information by assigning importance to words.


## Step 6: Train-Test Split

The dataset was divided into:

- Training Data
- Testing Data

This allows the model to learn from one portion of the dataset and evaluate performance on unseen data.


## Step 7: Machine Learning Model Training

Machine Learning models were trained on the processed dataset.

The best-performing model was selected based on evaluation metrics.


## Step 8: Model Evaluation

The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics were used to determine the performance of the model.

---

## Step 9: Save Model

After selecting the best-performing model, the following files were saved:

- best_model.pkl
- vectorizer.pkl

These files are used later inside the Streamlit application.


# Part 2 – Streamlit Web Application

The second part of the project focuses on deploying the trained machine learning model using Streamlit.

The application consists of three pages.


## Home Page

The Home page contains:

- Project Title
- Project Introduction
- Project Description
- Dataset Overview

This page gives users a brief understanding of the project.


## Data Overview Page

The Data Overview page displays:

- Dataset Preview
- Dataset Shape
- Class Distribution Graph
- Positive Review Word Cloud
- Negative Review Word Cloud

These visualizations help users understand the dataset.


## Sentiment Predictor Page

The Sentiment Predictor allows users to enter any Amazon review.

The application performs the following tasks:

- Accept user input
- Apply the same text preprocessing used during training
- Convert review into TF-IDF features
- Load the saved model
- Predict sentiment
- Display prediction result
- Display confidence score


# Project Workflow

Dataset
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Save Model
      │
      ▼
Streamlit Web App
      │
      ▼
User Review Prediction


# Project Folder Structure


Week5_NLP_Project/
│
├── app.py
├── best_model.pkl
├── vectorizer.pkl
├── DataSet (W5).csv
├── requirements.txt
├── README.md
└── Week5.ipynb

# Future Improvements

- Improve model accuracy using advanced NLP techniques.
- Add support for multilingual reviews.
- Deploy the application online using Streamlit Community Cloud.
- Add user authentication.
- Store prediction history in a database.


# Conclusion

This project demonstrates the complete workflow of a Natural Language Processing application, starting from data preprocessing and machine learning model development to deployment using Streamlit. The application allows users to classify Amazon reviews as Positive or Negative in real time using the trained model.


# Author

**Name:** *Manahil Zahra *

**Course:** AI\ML

**Task:** Week 5 NLP Task

**Tools Used:** Google Colab, VS Code, Streamlit, Python
