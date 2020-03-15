import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

encoder = LabelEncoder()


bank_data = pd.read_csv('bank-additional-full.csv', sep=',')


# print(bank_data.describe())
# print the rows by the columns..
# print(bank_data.shape)

# check whether its been loaded..
# print(bank_data.head)

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

# insert our target value (y) in target variable
y = bank_data['y'].values

# view target values
y[0:5]

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
# building and training the model
# create knn classifier
knn = KNeighborsClassifier(n_neighbors=3)

# fit the classifier into the data to train our model
knn.fit(X_train, y_train)
# testing the model
# print(knn.predict(X_test)[0:5])

# check accuracy of our model on the test data
print(knn.score(X_test, y_test))

# Using Cross validation

knn_cv = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

# train model with cv of 10
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
# print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))


plt.figure(figsize=(12, 10))
df = X
df['y'] = LabelEncoder().fit_transform(y)

cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

cor_target = abs(cor['y'])

rel_features = cor_target[cor_target > 0.2]
