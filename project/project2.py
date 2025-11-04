import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express  as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import  r2_score
from sklearn.linear_model import LinearRegression
data = pd.read_csv("housing_price_dataset.csv"  , usecols=['SquareFeet' , 'Price'])
print(data.shape)
print( data.isnull().sum())
print( data.info)



def train():
    df= data
    X = data[['SquareFeet']]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
    model= LinearRegression()
    model.fit(X_train , y_train)
    y_pred = model.predict(X_test)
    accuracy = r2_score( y_pred , y_test)
    print("  " , "accuracy :" , accuracy)
    return model

# we use differnt differnt model to increase accraucy like ( rf , dt  etc)

def main():
    st.title("linear regression house prediction app")
    st.write("put in your house size to know its price")
    model = train()
    size = st.number_input('house size' ,min_value= 1000 , max_value=4000 , value=1500)
    if  st.button('predict price'):
        predicted_price = model.predict([[size]])
        st.success(f"Estimated price: {round(predicted_price[0], 2)}")
        df = data
  
        fig = px.scatter(df,x="SquareFeet",y="Price",title="Square Feet vs House Price")

        fig.add_scatter(x=[size],y=[predicted_price[0]],
                        mode='markers',                    
                        marker=dict(size=15, color='red'),    
                        name='prediction')

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()




