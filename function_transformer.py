import numpy as np 
import pandas as pd 
import scipy.stats as   stats
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import FunctionTransformer , OneHotEncoder , LabelEncoder , StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Titanic.csv") 
print(df.head())
print(df.isnull().sum())
print(df.shape)

# plt.scatter(df.index , df['Age'])
# plt.show()

# print(df['Age'].mean())
# print(df['Age'].describe())

X= df[['Age' , 'Sex' , 'Fare']].copy()
y = df['Survived']

print(X.shape , y.shape)

print(X.isnull().sum())

X['Age']= X['Age'].fillna(X['Age'].mean())
print(X.isnull().sum())

# print(X.describe())

# convert categorical column into numercial using pandas 

"""X = pd.get_dummies(X , columns=['Sex'] ,dtype=int, drop_first=True)
print(X.head())"""

# convert categorical column into numerical using map function 

X['Sex'] = X['Sex'].map({'male' : 1 , 'female' : 0})
print(X.head())

# convert categorical column into numerical using OneHotEncoding  

# ohe = OneHotEncoder(drop='first' , dtype= np.int32)
# X_cat = ohe.fit_transform(X[['Sex']]).toarray()
# print(type(X_cat)) 
# X_num = X[['Age' , 'Fare']].values

# X= np.hstack((X_cat ,X_num))
# X= pd.DataFrame(X)


# x = X[['Age' , 'Fare']]
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print ( X_train)


plt.subplot(121)
sns.distplot(X_train['Age'])
plt.title("this is distplot")

plt.subplot(122)
stats.probplot(X_train['Age'], plot=plt)
plt.title("this is QQ plot")
plt.show()



plt.subplot(121)
sns.distplot(X_train['Fare'])   # output = right - skewed dataset 
plt.title("this is distplot")

plt.subplot(122)
stats.probplot(X_train['Fare'], plot=plt)
plt.title("this is QQ plot")
plt.show()


"""                applying transformation on data to increase the accuracy of model"""


trf = FunctionTransformer(func=np.log1p)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)






clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit( X_train_transformed , y_train)
clf2.fit(X_train_transformed , y_train)

y_pred = clf.predict(X_test_transformed)
y_pred1 = clf2.predict(X_test_transformed)

print("accuracy of LR:" , accuracy_score(y_test , y_pred))
print("accuracy of DT:" , accuracy_score(y_test , y_pred1))





