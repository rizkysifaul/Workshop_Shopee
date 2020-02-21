#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

#import the data
data = pd.read_csv("Data Clean.csv")
image = Image.open("house.png")
st.title("Welcome to the House Price Prediction App")
st.image(image, use_column_width=True)

#checking the data
st.write("This is an application for knowing how much range of house prices you choose using machine learning. Let's try and see!")
check_data = st.checkbox("See the simple data")
if check_data:
    st.write(data.head())
st.write("Now let's find out how much the prices when we choosing some parameters.")

#input the numbers
sqft_liv = st.slider("What is your square feet of living room?",int(data.sqft_living.min()),int(data.sqft_living.max()),int(data.sqft_living.mean()) )
bath     = st.slider("How many bathrooms?",int(data.bathrooms.min()),int(data.bathrooms.max()),int(data.bathrooms.mean()) )
bed      = st.slider("How many bedrooms?",int(data.bedrooms.min()),int(data.bedrooms.max()),int(data.bedrooms.mean()) )
floor    = st.slider("How many floor do you want?",int(data.floors.min()),int(data.floors.max()),int(data.floors.mean()) )

#splitting your data
X = data.drop('price', axis = 1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

#modelling step
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[sqft_liv,bath,bed,floor]])[0]

#checking prediction house price
if st.button("Run me!"):
    st.header("Your house prices prediction is USD {}".format(int(predictions)))
    st.subheader("Your range of prediction is USD {} - USD {}".format(int(predictions-errors),int(predictions+errors) ))