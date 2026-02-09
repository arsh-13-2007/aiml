# simpleimputer( univariate)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

data = pd.read_csv("Titanic.csv")
data  = data[['Cabin' ,'Age' , 'Survived']]
print(data.isnull().mean()*100)
X = data[['Cabin' ,'Age']]
y = data[['Survived']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# print( X_train , y_train )

trf = ColumnTransformer([
    ("imputing" , SimpleImputer(strategy='mean') , ['Age']) , 
    ('imputer' , SimpleImputer(strategy='most_frequent') , ['Cabin'])
],remainder='passthrough')
# data['Age'] = data['Age'].fillna(data['Age'].mean())
X_train = trf.fit_transform(X_train)
print(X_train)