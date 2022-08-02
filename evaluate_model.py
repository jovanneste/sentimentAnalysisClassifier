import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score

nb_model = pickle.load(open('nb_model.sav', 'rb'))
logr_model = pickle.load(open('logr_model.sav', 'rb'))

test_features = pickle.load(open('test_features.sav', 'rb'))
test_labels = pickle.load(open('test_labels.sav', 'rb'))

print("Evaluation of models using test set")

print("\nNaive Bayes Classifier Accuracy: "+ str(nb_model.score(test_features, test_labels)))
print("Logistic Regression Classifier Accuracy: "+ str(logr_model.score(test_features, test_labels))+"\n")
