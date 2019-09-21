#https://www.datacamp.com/community/tutorials/k-means-clustering-python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import threading

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

col_names=['passengerid','survived','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked']
train = pd.read_csv("../data-set/kaggle/titanic/train.csv", header=0, names=col_names)
print(train.head())
X=train[['passengerid','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked']]  # Features
y=train['survived']  # Labels

test_col_names=['passengerid','pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked']
test = pd.read_csv("../data-set/kaggle/titanic/test.csv", header=0, names=test_col_names)
print(test.head())

print(train.describe())

# To handle missing values
print(train.isna().sum())

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

print(test.isna().sum())

#To know about data
print(train[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False))

print(train[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False))

print(train[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False))

g = sns.FacetGrid(train, col='survived')
g.map(plt.hist, 'age', bins=20)

grid = sns.FacetGrid(train, col='survived', row='pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();

#plt.ion()

print(train.info())

train = train.drop(['name','ticket', 'cabin','embarked'], axis=1)
test = test.drop(['name','ticket', 'cabin','embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['sex'])
labelEncoder.fit(test['sex'])
train['sex'] = labelEncoder.transform(train['sex'])
test['sex'] = labelEncoder.transform(test['sex'])

Y=train['survived']
X=train.drop(['survived'],axis=1)

kmeans = KMeans(n_clusters=2) # You want cluster the passenger records into 2: Survived or Not survived
kmeans.fit(X)
y_pred=kmeans.predict(X)
print("Accuracy:",metrics.accuracy_score(Y, y_pred))

#The features in the dataset contain different ranges of values. So, what happens is a small change in a feature does not affect the other feature
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

best_random_state=0
max_accuracy=0
'''
for i in range(2000):
    if i%200==0:
        print(i)
    kmeans = KMeans(n_clusters=2, max_iter=600, init='k-means++',random_state=i)
    kmeans.fit(X_scaled)
    y_pred=kmeans.predict(X_scaled)
    if(max_accuracy<metrics.accuracy_score(Y, y_pred)):
        best_random_state=i
        max_accuracy=metrics.accuracy_score(Y, y_pred)
    #print("Accuracy:",metrics.accuracy_score(Y, y_pred)," For Random State: ",i)
'''
print("best_random_state= ",best_random_state)
kmeans = KMeans(n_clusters=2, max_iter=600, init='k-means++', random_state=best_random_state)
kmeans.fit(X_scaled)
y_pred=kmeans.predict(X_scaled)
print("Accuracy:",metrics.accuracy_score(Y, y_pred))
plt.show()