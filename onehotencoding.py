# one hot encoding is use to handle catogrical feature 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
# from sklearn.pipeline import Pipeline , make_pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder , LabelEncoder
from sklearn.model_selection  import train_test_split

data = pd.read_csv("cars.csv")
print(data.head())
print(data.isnull().sum())   # output = 0 
# print( data.duplicated().sum())
print(type(data))     # output = <class 'pandas.core.frame.DataFrame'>
print( data.shape)   # output : (8128, 5)

#   one of method to convert categorical column into numercial column ( encoding )   

# data_encoded  = pd.get_dummies( data  , columns=['fuel' , 'owner'], dtype=int , drop_first=True )    
# print(data_encoded.shape )   # output: (8128, 10)
# print(data_encoded.head())

X = data.iloc[: , : -1 ]
y = data.iloc[ :  , -1 ]
print( type(X) , type(y) )   # output <class 'pandas.core.frame.DataFrame'> <class 'pandas.core.series.Series'>
print( X.shape , y.shape)   # output (8128, 4) (8128,)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 )


#  one hot encoding    ( maximum time we use one hot encoding )
#  learn this process 
ohe = OneHotEncoder()
# X_cat = ohe.fit_transform (X_train[['fuel' , 'owner']]) 
# print( type(X_cat))  #  <class 'scipy.sparse._csr.csr_matrix'>
# print(X_cat.shape)   # (6502, 9)

X_train_cat = ohe.fit_transform (X_train[['fuel' , 'owner']] ).toarray()
print(X_train_cat)

# then add numercial and categorical columns again 
X_train_num= X_train[['brand','km_driven']].values    # use to convert into array 
print(type(X_train_num))

X_train_new = np.hstack(( X_train_num,X_train_cat ))

print(X_train_new)

X_train_new = pd.DataFrame(X_train_new)
print(X_train_new)

# apply same for test data

X_test_cat = ohe.fit_transform(X_test[['fuel' , 'owner']]).toarray()
X_test_num = X_test[['brand','km_driven']].values
X_test_new = np.hstack((X_test_num , X_test_cat))
print(X_test_new)
# then convert this comvert into pandas ( dataframe)
X_test_new = pd.DataFrame(X_test_new)
print(X_test_new.head())
print ( X_test_new.shape , X_train_new.shape) # output : (1626, 11) (6502, 11)


le = LabelEncoder()    # label encoding is use when we have more number of catgoraical value in single column 
X_test_new[0] = le.fit_transform(X_test_new[0])
X_train_new[0] = le.fit_transform(X_train_new[0])

print(X_test_new.head())  
print(X_train_new.head())  