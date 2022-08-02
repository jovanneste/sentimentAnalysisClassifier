import pickle
import spacy
import numpy as np
import pandas as pd

from tkinter import *

nlp = spacy.load('en_core_web_sm', disable=["ner"])
nlp.remove_pipe('tagger')
nlp.remove_pipe('parser')


def text_pipeline_spacy(text):
    tokens = []
    doc = nlp(text)
    for t in doc:
        if not t.is_stop and not t.is_punct and not t.is_space:
            tokens.append(t.lemma_.lower())
    return tokens


def predict(sentence):
    nb_model = pickle.load(open('nb_model.sav', 'rb'))
    logr_model = pickle.load(open('logr_model.sav', 'rb'))
    one_hot_vectorizer = pickle.load(open('one_hot_vectorizer.sav', 'rb'))

    sentence_features = one_hot_vectorizer.transform([sentence])

    predictionLR = int(logr_model.predict(sentence_features))
    predictionNB = int(nb_model.predict(sentence_features))

    prediction = (predictionLR+predictionNB)//2

    if prediction==0:
        return "Negative sentiment"
    elif prediction==4:
        return "Positive sentiment"
    else:
        return "Neutral sentiment"


window = Tk()

window.title("Sentiment Analysis Predictor")
window.geometry('450x300')

lbl1 = Label(window, text="Enter sentence below", font=("Arial", 15))
lbl1.grid(column=100, row=0)

lbl2 = Label(window, text="", font=("Arial", 20))
lbl2.grid(column=100, row=350)

txt = Entry(window,width=50)
txt.grid(column=100, row=50)

def clicked():
    sentence = txt.get()
    predicted_sentiment = predict(sentence)
    print(predicted_sentiment)
    lbl2.configure(text=predicted_sentiment)

del_btn = Button(window, text='Clear', command=lambda: txt.delete(0, END))
del_btn.grid(column=100, row=450)

btn1 = Button(window, text="Okay", command=clicked)
btn1.grid(column=100, row=150)


window.mainloop()
