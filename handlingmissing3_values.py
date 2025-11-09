import numpy as np 
import pandas as pd
from sklearn.metrics import precision_score , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer , KNNImputer

data = pd.read_csv("Titanic.csv")
print( data.head())
print( data.isnull().sum())
print( data.isnull().mean()*100)

X = data[['Age' , 'Pclass' ,'Fare']]
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
knn = KNNImputer(n_neighbors = 10)
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.fit_transform(X_test)

# si = SimpleImputer(strategy='mean')
# X_test_trf = si.fit_transform(X_test)
# X_train_trf = si.fit_transform(X_train)


lr = LogisticRegression()
lr.fit(X_train_trf , y_train)
y_pred = lr.predict(X_test_trf)
accuracy = accuracy_score(  y_test, y_pred)
precision = precision_score(y_test , y_pred)
print( accuracy , precision)