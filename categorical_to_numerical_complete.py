import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from  sklearn.preprocessing import OneHotEncoder , LabelEncoder , OrdinalEncoder , StandardScaler
from sklearn.model_selection import train_test_split
le = LabelEncoder()


df = pd.read_csv("loan_project6.csv")

print(df.head())
print( df.isnull().sum())

df = df.dropna(subset=['Gender' , 'Married'])

print(df.shape)
 
X = df[['Gender' , 'Married' , 'Education' , 'Property_Area' , 'ApplicantIncome']]
y= df['Loan_Status']

# print(X.shape , y.shape)
# print(X.isnull().sum())
# print(y.isnull().sum())

print(X.head())
print(y)
"""                                               ordinal encoding            """


# oe = OrdinalEncoder(categories=[['Not Graduate', 'Graduate']])

# X['Education'] = oe.fit_transform(X[['Education']])


""" convert categorial data into numercial using pandas """

# X_encoded = pd.get_dummies(X, columns=['Gender' , 'Married' , 'Education' , 'Property_Area'] , dtype=int , drop_first=True)
# print(X_encoded)


""" convert categorial data into numercial using map function  """

# X['Gender' ] = X['Gender'].map({"Male" : 1 , "Female" : 0 }) 
# print(X)



"""                            one hot encoding     """

ohe = OneHotEncoder(drop='first') ; 

X_cat = ohe.fit_transform(X[['Gender' , 'Married','Education','Property_Area']] ).toarray()
 
print(type(X_cat))
X_num = X[['ApplicantIncome']].values
print(type(X_num))

X = np.hstack((X_cat , X_num))
print(type(X))

X= pd.DataFrame(X)
print(type(X))
print(X.head())
""" label encoding"""

y = le.fit_transform(y)   # label encoding is only applies on output column not input columns (important)

print(type(y))
y= pd.DataFrame(y)
print(type(y))
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

