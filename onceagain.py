import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from  sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.preprocessing import StandardScaler , OneHotEncoder, LabelEncoder 
from sklearn.impute import SimpleImputer
ss = StandardScaler()
model = LogisticRegression()


df = pd.read_csv("placement.csv")
print(df.head())

print(df.isnull().sum()) 
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
# plt.boxplot(df[['cgpa' ,'placement_exam_marks']])      //box plot is use for check outliers in data or not 
# plt.show()

X = df.iloc[: , :-1 ]
y = df['placed'] 
# print(X.head())
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.fit_transform(X_test)

print(type(X_train_scaled)) # <class 'numpy.ndarray'>
print(type(X_test_scaled)) # <class 'numpy.ndarray'>


model.fit(X_train_scaled , y_train)
y_pred = model.predict(X_test_scaled)

print(accuracy_score( y_test, y_pred))

