# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix


encoder = LabelEncoder()


bank_data = pd.read_csv('bank-additional-full.csv', sep=',')

bank_data['marital'].replace('unknown', 'married', inplace=True)
bank_data['default'].replace('unknown', 'no', inplace=True)
bank_data['loan'].replace('unknown', 'no', inplace=True)


bank_data.drop(bank_data[bank_data.housing == 'unknown'].index, inplace=True)
bank_data.drop(bank_data[bank_data.education == 'unknown'].index, inplace=True)
bank_data.drop(bank_data[bank_data.job == 'unknown'].index, inplace=True)
bank_data.drop(bank_data[bank_data.age >= 70].index, inplace=True)

bank_data.describe(include='all')
# print(bank_data.shape)
# split dataset into inputs and target (Our class that we are looking for, in this case = y)
X = bank_data.drop(columns=['y'])
# encoding the data from strings to numbers
X.job = encoder.fit_transform(X.job)
X.marital = encoder.fit_transform(X.marital)
X.education = encoder.fit_transform(X.education)
X.default = encoder.fit_transform(X.default)
X.housing = encoder.fit_transform(X.housing)
X.contact = encoder.fit_transform(X.contact)
X.loan = encoder.fit_transform(X.loan)
X.month = encoder.fit_transform(X.month)
X.day_of_week = encoder.fit_transform(X.day_of_week)
X.poutcome = encoder.fit_transform(X.poutcome)

# confirm target variable has been removed --> should print out one less column
X.head()

# pca_bank = PCA(n_components=2)
# principalComponentBank = pca_bank.fit_transform(X)

# # dataframe to have all the principal component values
# principal_bank_Df = pd.DataFrame(data=principalComponentBank, columns=[
#                                  'principal component 1', 'principal component 2'])
# print(principal_bank_Df.tail())

# # variance each component has

# print('Explained variation per principal component: {}'.format(
#     pca_bank.explained_variance_ratio_))

# print(np.mean(X), np.std(X))

# insert our target value (y) in target variable
y = bank_data['y'].values

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
# print(X_train.shape)
# print(X_test.shape)

scores = cross_val_score(MLPClassifier(hidden_layer_sizes=(19, 19, 19),
                                       activation='relu', solver='adam', max_iter=500), X, y, cv=10, n_jobs=-1)

print(scores.mean())

# building the neural network model
# model set to 3 layers & neurons = same no of features (19)

# mlp = MLPClassifier(hidden_layer_sizes=(19, 19, 19),
#                     activation='relu', solver='adam', max_iter=500)
# mlp.fit(X_train, y_train)

# # generate predictions on train and model set
# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)

# # Evaluate performance of model on training data
# print(confusion_matrix(y_train, predict_train))
# print(classification_report(y_train, predict_train))

# # Evaluate performance of model on test data
# print(confusion_matrix(y_test, predict_test))
# print(classification_report(y_test, predict_test))
