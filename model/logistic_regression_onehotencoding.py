# logistic regression is a classification techinque 
# classification techinque is when  output or y is dependent on X 

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.preprocessing import  OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
Lr = LogisticRegression()
data = sns.load_dataset('iris')
print(data.isnull().sum())

print( data['species'].unique())
data = data.drop(data[data['species'] == 'setosa'].index)
print( data.head())

#               how to convert catogerical column into numerical 
# data = pd.get_dummies(data , columns=["species"] , dtype=int , drop_first=True )    # create multiple columns   
data['species'] = data['species'].map({'versicolor':0  ,'virginica': 1} ) 
#       # good method  
#label encoding 
# encoding = OneHotEncoder(dtype=int)
# data_encoded= encoding.fit_transform(data[['species']])
# one hot encoding 
print( data)
X = data.iloc[:, :-1] 
y = data.iloc[: , -1 ]
print( X.head() , y.head())
print( y )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
parameters= {'penalty':['l1','l2','elasticnet'] , 'C':[1,2,3,4,5,6,10,20,30,40,50] ,'max_iter' :[100,200,300 ] }
classifier_regressor = GridSearchCV( Lr , param_grid=parameters , scoring='accuracy' , cv= 5 )
classifier_regressor.fit(X_train   , y_train)
print( classifier_regressor.best_params_)
print( classifier_regressor.best_score_)
y_pred= classifier_regressor.predict(X_test)
precision = precision_score( y_pred , y_test)
print( precision)
