from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression # model 
from sklearn import set_config # it is use to visualize the pipeline  

# steps= [('standardscaler' , StandardScaler()) , 'classifier' , LogisticRegression()]
# print( steps)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
# pipe = Pipeline( steps)  # pipeline is sequence apply list of transforms and a final estimator 
set_config( display='diagram')
print( pipe)
X, y = make_classification(return_X_y =True , n_samples=1000 , n_features=20  )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")
pipe.fit(X_train, y_train)
y_predicted = pipe.predict( X_test)# we not do standard scaling seperatly usinig pipe we do directly on both 
print( y_predicted)