import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

model: Pipeline = joblib.load('classifier.gz')


def score(args):
    X = pd.DataFrame([args])
    return model.predict_proba(X)[0][1]
