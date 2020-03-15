# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv("winequality-red.csv")

X = data.drop(columns=['quality'])
y = data.quality

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

model = SVC(gamma='auto', random_state=42)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)
score
