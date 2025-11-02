import numpy as np 
import seaborn as sns 
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV   # hyperparameter tuning 
df =sns.load_dataset('tips')
print(df.head())

X=df.iloc[: ,1:]
y= df['total_bill']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print( df.isnull().sum())
numercial_preprocessor = Pipeline([
    ("imputing" , SimpleImputer( missing_values=np.nan, strategy= 'mean')) , 
    ("scaler" , StandardScaler())
])

categorical_preprocessor= Pipeline([
    ("imutation_constant" , SimpleImputer( fill_value="missing", strategy="constant")) , 
    ("scaling" , StandardScaler())
])

print(numercial_preprocessor , categorical_preprocessor)