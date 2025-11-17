import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics  import  accuracy_score
from sklearn.compose import ColumnTransformer


df = pd.read_csv("loan_project6.csv")
print(df.head())
print(df.isnull().sum())
# sns.pairplot(df)
# plt.show()
# sns.countplot(x='Loan_Status', data=df)
# plt.show()
df['Loan_Status'] = df['Loan_Status'].map({ "Y" : 1 , "N" : 0 } )
print( df.head())

print(df['Credit_History'].value_counts())

sns.countplot(x= df['Loan_Status'] , hue='Credit_History', data=df)
plt.show()
print(type(df))
print(df.shape)
df = df.dropna()
print(type(df))
print(df.shape)
# print(df.isnull().sum())
# df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
# print(df.isnull().sum())

# onehot encoding 


# cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed' , 'Property_Area' , 'Education'] # your categorical cols

# ct = ColumnTransformer(
#     transformers=[
#         ('ohe', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
#     ],
#     remainder='passthrough'
# )

# df_encoded = ct.fit_transform(df)

# df = pd.DataFrame(df_encoded)
# print(df)

# labelencoding 

le = LabelEncoder()

cols = ['Gender','Married','Dependents','Self_Employed','Property_Area','Education']

for col in cols:
    df[col] = le.fit_transform(df[col])

print(df.head())

# get dummines 

# df_encoded = pd.get_dummies(df , columns=['Gender', 'Married', 'Dependents', 'Self_Employed' , 'Property_Area' , 'Education'] , dtype=int)
# print(df_encoded)

X = df.drop(columns=['Loan_Status' , 'Loan_ID'] , axis=1 )
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

ss = StandardScaler()

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

classifier = SVC(kernel='linear')
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



