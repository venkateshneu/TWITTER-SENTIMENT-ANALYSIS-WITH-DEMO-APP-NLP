import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv("twitter_training.csv", header=None)
    df.columns = ['text_id', 'product', 'sentiment', 'text']
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df = df.replace({"Irrelevant": "Neutral"})
    return df

# Preprocess text data
def preprocess_text(text):
    # Add your preprocessing steps here
    return text

# Sentiment analysis using SVM
def sentiment_analysis(df, product_name):
    product_df = df[df['product'] == product_name]
    product_df['cleaned_text'] = product_df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        product_df['cleaned_text'], product_df['sentiment'], test_size=0.2, random_state=42)

    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', SVC(kernel='linear', probability=True))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit app
st.title("Sentiment Analysis and Topic Modeling")

# Sidebar for selecting options
st.sidebar.header("Select Options")
product_list = ['Overwatch', 'Amazon', 'Facebook', 'Verizon', 'Google']
selected_product = st.sidebar.selectbox("Select a Product", product_list)
nlp_task = st.sidebar.selectbox("Choose an NLP Task", ['Sentiment Analysis'])

# Sentiment Analysis
if nlp_task == 'Sentiment Analysis':
    st.subheader("Sentiment Analysis for Selected Product")
    st.write(f"Selected Product: {selected_product}")

    df = load_data()
    model, accuracy = sentiment_analysis(df, selected_product)
    st.write(f"Accuracy: {accuracy:.4f}")

    # Add input box for user review
    user_review = st.text_input("Enter a review for sentiment prediction:")
    if st.button("Predict Sentiment"):
        if user_review:
            prediction = model.predict([user_review])
            
            # Highlight the prediction result
            if prediction[0] == 'Positive':
                st.success(f"The predicted sentiment for the review is: **{prediction[0]}** 🎉")
            elif prediction[0] == 'Negative':
                st.error(f"The predicted sentiment for the review is: **{prediction[0]}** 😞")
            else:
                st.warning(f"The predicted sentiment for the review is: **{prediction[0]}** 🤔")
        else:
            st.write("Please enter a review to predict.")

    st.write("You can implement additional evaluations here.")
