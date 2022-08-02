import pandas as pd
import pickle
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

print("Loading data...")

data = pd.read_csv('training.1600000.processed.noemoticon.csv')

user_n = int(input("How many tweets (max 1.6M): "))

data = data.sample(n=user_n)
data.columns = ['target','id','date','flag','user','text']



# saving the dataframe
data.to_csv('data.csv', index=False)

p_train = float(input("Training data split (0-1): "))
random_training_data = data.sample(frac=1)
train_split = int(len(random_training_data)*p_train)

train_data = random_training_data.iloc[:train_split, :]
test_data = random_training_data.iloc[train_split:, :]

print('Train set contains {:d} tweets.'.format(len(train_data)))
print('Test set contains {:d} tweets.'.format(len(test_data)))



def display_info(n, label):
    print('Train set contains ' + str(100*(sum(train_data['target'] == n))/len(train_data)) + ' ' +label+' tweets')
    print('Test set contains ' + str(100*(sum(test_data['target'] == n))/len(test_data)) + ' ' +label+' tweets')

display_info(0, 'negative')
display_info(4, 'positive')


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


print("Training model...")

one_hot_vectorizer = CountVectorizer(tokenizer=text_pipeline_spacy, binary=True)
print("Training features")
train_features = one_hot_vectorizer.fit_transform(train_data['text'])
train_labels = train_data['target']
print("Testing features")
test_features = one_hot_vectorizer.transform(test_data['text'])
test_labels = test_data['target']

print("Dumping...")
pickle.dump(test_labels, open('test_labels.sav', 'wb'))

pickle.dump(train_features, open('train_features.sav', 'wb'))
pickle.dump(train_labels, open('train_labels.sav', 'wb'))
pickle.dump(test_features, open('test_features.sav', 'wb'))
pickle.dump(test_labels, open('test_labels.sav', 'wb'))

bayes_classifier = BernoulliNB()
nb_model = bayes_classifier.fit(train_features, train_labels)
pickle.dump(nb_model, open('nb_model.sav', 'wb'))

logr = LogisticRegression(solver='saga')
logr_model = logr.fit(train_features, train_labels)
pickle.dump(logr_model, open('logr_model.sav', 'wb'))

pickle.dump(one_hot_vectorizer, open('one_hot_vectorizer.sav', 'wb'))

print("Model trained (LR and Bayes)")
