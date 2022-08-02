# sentimentAnalysisClassifier

Trains two models on a given subset of the 1.6 million sentiment tweet [dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). Can then input a given sentence and see its predicted sentiment.

## Deployment

CLI deployment:
- python train_model.py (trains logistic regression and naive Bayes models)
- python evaluate_model.py (evalutes models on the some test data not used in the training)
- python predict.py (launches GUI to obtain user input)
