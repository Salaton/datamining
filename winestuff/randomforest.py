# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline  # Cross validation
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score  # Evaluation
from sklearn.externals import joblib  # For model persistence

data = pd.read_csv("winequality-red.csv")

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

y = data.quality
X = data.drop(columns=["quality"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


model.fit(X_train, y_train)
model.score(X_test, y_test)


# %%

export_graphviz(model.estimators_[0], out_file="redwine.dot")


# %%
# Hyperparameter tuning with RandomizedSearchCV

opt_model = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=400, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]  # Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

opt_model_cv = RandomizedSearchCV(
    opt_model, random_grid, n_iter=100, cv=8, n_jobs=-1)
opt_model_cv.fit(X_train, y_train)
opt_model_cv.score(X_test, y_test)
