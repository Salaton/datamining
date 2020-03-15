# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd

data = pd.read_csv("winequality-red.csv")

X = data.drop(columns=['quality'])
y = data.quality

model = DecisionTreeClassifier(random_state=42)

scores = cross_val_score(model, X, y, cv=8)
scores.mean()


# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

model.fit(X_train, y_train)
columns = [col for col in X.columns]

export_graphviz(model, out_file="redwinetree.dot", feature_names=columns)

model.score(X_test, y_test)
