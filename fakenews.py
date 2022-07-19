import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def process(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english')and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('modelf.pkl','rb'))

st.title(" Fake News Classifier")
input=st.text_input("Enter your News")
if st.button('Predict'):
    news=process(input)
    vn=cv.transform([news])
    r=model.predict(vn)[0]
#1:fake ,0:true
    if r==1:
        st.header("It is a fake News")
    else:
        st.header("It is a reliable news")
