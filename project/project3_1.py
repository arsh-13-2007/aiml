import streamlit as st
import pickle
import string 
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text( text):
    text = text.lower()
    text = nltk.word_tokenize( text)  # it  is use to split into single single word 
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y [:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i  in text:
        y.append(ps.stem(i))

    return " ".join(y)

mnb = pickle.load(open("model.pkl", 'rb'))
cv = pickle.load(open("vectorizer.pkl", 'rb'))

st.title("Email.SMS spam classifier")

input_sms = st.text_input("enter msg")
if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = cv.transform([transformed_sms])
    
  
    result = mnb.predict(vector_input)[0]
    
  
    if result == 1:
        st.header("🚨 Spam Message!")
    else:
        st.header("✅ Not Spam")
