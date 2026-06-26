import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import json
import os

train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

train_data = train_data.rename(columns={'default payment next month': 'default'})
test_data = test_data.rename(columns={'default payment next month': 'default'})
train_data.drop(columns=['ID'], inplace=True, errors='ignore')
test_data.drop(columns=['ID'], inplace=True, errors='ignore')
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
train_data = train_data[(train_data["EDUCATION"] != 0) & (train_data["MARRIAGE"] != 0)]
test_data = test_data[(test_data["EDUCATION"] != 0) & (test_data["MARRIAGE"] != 0)]
train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

x_train, y_train = train_data.drop(columns=['default']), train_data['default']
x_test, y_test = test_data.drop(columns=['default']), test_data['default']
categorical_features = x_train.select_dtypes(include=['object', 'category']).columns.tolist()

transformer = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
pipeline = Pipeline([
    ('preprocessor', transformer),
    ('scaler', MinMaxScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression(max_iter=500, random_state=42))
])

params = {
    'feature_selection__k': range(1, 11),
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear'],
    "classifier__max_iter": [100, 200]
}
grid_search = GridSearchCV(pipeline, param_grid=params, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True)
grid_search.fit(x_train, y_train)

os.makedirs("files/models/", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", 'wb') as f:
    pickle.dump(grid_search, f)

def compute_metrics(y_true, y_pred, dataset):
    return {
        'type': 'metrics',
        'dataset': dataset,
        'precision': precision_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def confusion_matrix_data(y_true, y_pred, dataset):
    cm = confusion_matrix(y_true, y_pred)
    return {
        'type': 'cm_matrix',
        'dataset': dataset,
        'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

metrics = [
    compute_metrics(y_train, grid_search.predict(x_train), 'train'),
    compute_metrics(y_test, grid_search.predict(x_test), 'test'),
    confusion_matrix_data(y_train, grid_search.predict(x_train), 'train'),
    confusion_matrix_data(y_test, grid_search.predict(x_test), 'test')
]

os.makedirs("files/output/", exist_ok=True)
with open("files/output/metrics.json", "w") as f:
    for metric in metrics:
        f.write(json.dumps(metric) + "\n")