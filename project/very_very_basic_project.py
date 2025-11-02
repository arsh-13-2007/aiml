import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from  sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics  import accuracy_score
global_temp = pd.read_csv("weather_dataset.csv")
print( global_temp.info() )
def clean_data(df):
    df = df.copy()
    df= df.drop(columns=["LandAverageTemperatureUncertainty" , 
                         "LandMaxTemperatureUncertainty" ,
                         "LandMinTemperatureUncertainty",
                         "LandAndOceanAverageTemperatureUncertainty"])
    
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce", infer_datetime_format=True, dayfirst=True)
    df = df.dropna(subset=["dt"])
    df["month"] = df["dt"].dt.month
    df["year"]= df["dt"].dt.year
    df = df.drop(columns=["dt"] , axis= 1 )
    df = df.dropna()
    df = df.drop(columns=["month"] , axis = 1 )
    df = df.set_index(["year"])
    return df 

global_temp = clean_data( global_temp)
print(global_temp.head())
sns.heatmap(global_temp.corr() , annot= True)
plt.show()
X = global_temp.iloc[: , :-1]
y= global_temp.iloc[: , -1]
print(X.head() , y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

forest = Pipeline([
    ("kbest" , SelectKBest(k="all")),
    ("scaler" , StandardScaler()) , 
    ("randomforest" , RandomForestRegressor(n_estimators=100, random_state=77 , n_jobs=1 ,max_depth=50)) 
])
forest.fit(X_train , y_train)
answer = forest.predict(X_test)
print( answer)

