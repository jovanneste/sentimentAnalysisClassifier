import pickle
import spacy
import numpy as np
import pandas as pd

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

nb_model = pickle.load(open('nb_model.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))
one_hot_vectorizer = pickle.load(open('one_hot_vectorizer.sav', 'rb'))


sentence = input("Enter sentence: ")
sentence_features = one_hot_vectorizer.transform([sentence])

predictionLR = logr_model.predict(sentence_features)
predictionNB = nb_model.predict(sentence_features)

print("LR prediction: " + str(predictionLR))
print("NB prediction: " + str(predictionNB))
