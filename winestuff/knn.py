# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

wine_data = pd.read_csv('winequality-red.csv')

X = wine_data.iloc[:, :-1]
y = wine_data.iloc[:, 11]

knn = KNeighborsClassifier(n_neighbors=5)

# Using holdout method
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y)

knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# %%
# Using folds (10)
knn_cv = KNeighborsClassifier(n_neighbors=5)

cv_scores = cross_val_score(knn_cv, X, y, cv=10)
print(cv_scores)
print(f"cv_scores mean:{np.mean(cv_scores)}")


# %%
# Using hypertuned parameters

param_grid = {'n_neighbors': np.arange(5, 50)}

knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)

knn_gscv.fit(X, y)
knn_gscv.best_score_


# %%
# Standardizing
knn_std = KNeighborsClassifier()

scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

knn_std.fit(X_train_std, y_train)

knn_std.score(X_test_std, y_test)


# %%
# Standardized with hypertuned params
param_grid = {'n_neighbors': np.arange(5, 50)}

knn_std_gscv = GridSearchCV(KNeighborsClassifier(),
                            param_grid, cv=10, n_jobs=-1)

scaler2 = StandardScaler()
scaler2 = scaler2.fit(X)
X_std = scaler2.transform(X)

knn_std_gscv.fit(X_std, y)
knn_std_gscv.best_score_


# %%
# Selecting best 4 features

X_reduced = SelectKBest(chi2, k=5).fit_transform(X, y)
X_reduced_train, X_reduced_test, y_red_train, y_red_test = train_test_split(
    X_reduced, y, random_state=42, test_size=0.25, stratify=y)

knn_red = KNeighborsClassifier(n_neighbors=5)
knn_red.fit(X_reduced_train, y_red_train)
print(knn_red.score(X_reduced_test, y_red_test))

knn_red_gscv = GridSearchCV(KNeighborsClassifier(
), {'n_neighbors': np.arange(2, 70)}, cv=10, n_jobs=-1)
knn_red_gscv.fit(X_reduced_train, y_red_train)
knn_red_gscv.score(X_reduced_test, y_red_test)


# %%
# Using features selected by weka
X_weka = X.drop(columns=["fixed acidity", "citric acid", "residual sugar",
                         "free sulfur dioxide", "total sulfur dioxide", "density", ])

X_weka_train, X_weka_test, y_weka_train, y_weka_test = train_test_split(
    X_weka, y, random_state=42, stratify=y)
knn_weka = KNeighborsClassifier(n_neighbors=14)
knn_weka.fit(X_weka_train, y_weka_train)

knn_weka.score(X_weka_test, y_weka_test)


# %%
# Using hypertuned parameters and reduced features

param_grid = {'n_neighbors': np.arange(5, 70)}

knn_gscv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)

knn_gscv.fit(X_weka, y)
knn_gscv.best_score_


# %%
# Finding important attributes using Pearson Correlation

plt.figure(figsize=(12, 10))
df = X
df['quality'] = LabelEncoder().fit_transform(y)

cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor['quality'])

rel_features = cor_target[cor_target > 0.2]


# %%
# Best features standardized
knn_best = KNeighborsClassifier()
scaler = StandardScaler()

X_best = X[['volatile acidity', 'citric acid', 'sulphates', 'alcohol']]
X_best_std = scaler.fit_transform(X_best)

knn_best_gscv = GridSearchCV(knn_best, param_grid, cv=10, n_jobs=-1)

knn_best_gscv.fit(X_best_std, y)

knn_best_gscv.best_score_
