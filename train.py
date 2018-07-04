import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from model_features import ModelFeatures
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

df = pd.read_csv('data/train.csv')

y = df['Survived']
X = df.drop(['Survived'], axis=1)

model = make_pipeline(
    ModelFeatures(),
    Imputer(),
    RandomForestClassifier()
)

model.fit(X, y)

y_hat = model.predict(X)

print('-- confusion matrix --')
print(confusion_matrix(y, y_hat))

joblib.dump(model, 'classifier.gz', compress=3, protocol=3)
