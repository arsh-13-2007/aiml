import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
ss= StandardScaler()



data = pd.read_csv("heart_dataset.csv" ,)
# print( data.head())
# print(data['target'].head())   # this sntax is use to print particular column of dataset 
# print( data.shape)
# print(data.isnull().sum())
X = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']]
y = data['target']   # we not do feature scling on trageted data because it is outputed data not input 

X_scaled= ss.fit_transform(X)
X= X_scaled
print( X, y )
#                                    in place of simply use train , test split we use cross-validation  techinques like ( loocv , k fold, etc.) 


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify= y )

models = [ LogisticRegression() , SVC(kernel='linear') , KNeighborsClassifier() , RandomForestClassifier()]
# def campare_models(models):
#     for model in models:
#         model.fit(X_train , y_train)
#         test_data_prediction = model.predict(X_test)
#         accuracy = accuracy_score( y_test , test_data_prediction)
#         print('accuracy of', model , ' = ' , accuracy) 

# campare_models(models)

# logisticRegression perform best in this dataset 



#                                     cross validation ( k folds )

def campare(models):
    for model in models:
        cv_model = cross_val_score( model , X, y , cv=5 )  #  k in k-fold method  
        mean = sum( cv_model)/ len(cv_model)
        print("accuracy " , model , ' = ' ,round(mean*100 , 2)  )  


campare(models)