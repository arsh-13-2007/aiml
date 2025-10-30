import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression
classification = LogisticRegression()
from sklearn.impute import SimpleImputer
si = SimpleImputer( strategy= 'median') 

data = pd.read_csv("Titanic.csv" , usecols=['Pclass' , 'Age' , 'Survived' ,'Fare']) # learn to select some important features from big dataset 
print( data.head())  
data['Age'] = si.fit_transform(data[['age']])
# data['Age'] = data['Age'].fillna(data.Age.median())
X = data.iloc[ : ,1:]
y = data.iloc[: , 0]
print( X.head())
print( y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
print( X_test , X_train)
X_train_scaled = scaler.fit_transform(X_train)    # in preprocessing for train we  do fit and transfrom both 
X_test_scaled = scaler.transform(X_test)    # in preprocessing for test we do only transform 
print(X_train_scaled , X_test_scaled)
# in model   fit is used for training and predict is used for testing the data 
classification.fit(X_train_scaled , y_train)
LogisticRegression()
print(classification.predict( X_test_scaled))







