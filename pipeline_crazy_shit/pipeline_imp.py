import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , MinMaxScaler , OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score , precision_score
from sklearn import set_config

df = pd.read_csv("Titanic.csv")
df.drop(columns=['PassengerId' , 'Name' , 'Ticket' , 'Cabin'] , inplace=True )
print( df.head())
X= df.iloc[: , 1: ]
y = df['Survived']
print( X.shape  , y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print ( X_train)
print(df.isnull().sum())

trf1 = ColumnTransformer([
    ('imputing' , SimpleImputer(strategy='mean') , [2]) ,
     ('impute_categorical' , SimpleImputer(strategy='most_frequent') , [6] ) 
], remainder='passthrough')

trf2 = ColumnTransformer([
    
    ('ohc' , OneHotEncoder(sparse_output=False, handle_unknown='ignore') ,[1,3])
] , remainder='passthrough')

trf3 = ColumnTransformer([
    ('scaling' , StandardScaler() , slice(0, 10))
] , remainder='passthrough')

trf4 = DecisionTreeClassifier()

pipe = Pipeline([
    ('trf1' , trf1) ,
    ('trf2' , trf2) ,
    ('trf3' , trf3) ,
    ('model' , trf4)
])
set_config(display='diagram')
plt.show()
pipe.fit(X_train , y_train)
y_pred = pipe.predict(X_test)

accuracy = accuracy_score(  y_test , y_pred )
precision = precision_score(y_test , y_pred )
print(accuracy)
print(precision)