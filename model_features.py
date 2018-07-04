import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.pipeline import TransformerMixin


class ModelFeatures(TransformerMixin):
    """
    Filter columns to keep for prediction and process categorical columns
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
        X['Pclass'] = X['Pclass'].astype('object')

        # tuples of (ordered, categories)
        categorical = {
            'Pclass': (True, [1, 2, 3]),
            'Sex': (False, ['male', 'female'])
        }

        for c in X.columns:
            if c in categorical:
                t = CategoricalDtype(categories=categorical[c][1], ordered=categorical[c][0])
                X[c] = X[c].astype(t)

        # convert ordered categorical columns to integer column and create dummy columns for unordered
        for c in X.columns:
            if X[c].dtype.name == 'category':
                if X[c].cat.ordered:
                    X[c] = X[c].cat.codes

        return pd.get_dummies(X)
