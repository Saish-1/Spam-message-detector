"""
1. Preprocess
2. Vectorize
3. Predict
4. Display
"""

import streamlit as st
import joblib
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string 

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model.pkl")

st.header("Spam Classifier")
st.markdown("------------")

input_sms = st.text_input("Enter the mail/message to check if it is a spam mail: ")
button = st.button("Predict")
if button:
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.write("SPAM!")
    else:
        st.write("Not Spam")
