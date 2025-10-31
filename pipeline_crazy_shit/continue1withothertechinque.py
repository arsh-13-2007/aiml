import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.svm import SVC 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer  # very very very improtant  in point of view of pipeline
from sklearn.pipeline import make_pipeline 
X, y = make_classification(return_X_y=True )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
pipe = Pipeline([
    ("scaling" , StandardScaler ()) ,
    ("pca" , PCA(n_components=3)) ,
    ("svc" , SVC())
    ])
#                         very very important
# if we have any imputer 
# pipeline is look like 

# pipe = Pipeline([
#     ("imputing" ,SimpleImputer(missing_values=np.nan  , strategy = "mean"))     # this is for numerical  columns  , 
#     ("scaling" , StandardScaler ()) ,
#     ("pca" , PCA(n_components=3)) ,
#     ("svc" , SVC())
# ])


#                                  


print(pipe)
print( X_train)
pipe.fit( X_train , y_train)
y_predict = pipe.predict( X_test)
print ( y_predict )


# for categorical features we use startegy = "most_frequent" or  startegy = "missing"


numerical_featuer=Pipeline([
    ("imputing" ,SimpleImputer(missing_values=np.nan  , strategy = "mean")) ,    
    ("scaling" , StandardScaler ())
])


categorical_feature = Pipeline([
    ("inputing_categorical" , SimpleImputer(fill_value="missing" , strategy="missing")) , 
     ("one_hot_encoding",OneHotEncoder( handle_unknown="ignore") )
])



preprocessing = ColumnTransformer([
    ("categorical_feature" , categorical_feature,["gender" , "city"]) , 
    ("numerical_featuer" , numerical_featuer , ["age" , "height"])
])

pipe = make_pipeline(preprocessing ,SVC() )
print(pipe)