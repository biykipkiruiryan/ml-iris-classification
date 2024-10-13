import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
print(sns.__version__)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.title('Iris Flower Classification')

st.write("This app classifies Iris flowers into different species based on their features.")

st.header('Choose your parameters')

# Reading and preparing the dataset
Dataframe = pd.read_csv('Iris.csv', header=None)
Dataframe.rename(columns={
    0: 'sepal length (cm)', 
    1: 'sepal width (cm)',
    2: 'petal length (cm)',
    3: 'petal width (cm)',
    4: 'Class'}, inplace=True)

# Slider for user input
sepal_length = st.sidebar.slider('sepal length (cm)', 4.30, 7.90, 5.84)
sepal_width = st.sidebar.slider('sepal width (cm)', 2.00, 4.40, 3.05)
petal_length = st.sidebar.slider('petal length (cm)', 1.00, 6.90, 3.76)
petal_width = st.sidebar.slider('petal width (cm)', 0.10, 2.50, 1.20)

# User input DataFrame
user_input = {'sepal length (cm)': sepal_length,
              'sepal width (cm)': sepal_width,
              'petal length (cm)': petal_length,
              'petal width (cm)': petal_width}

user_data = pd.DataFrame(user_input, index=[0])

# Splitting the data into X (features) and Y (labels)
X = Dataframe.drop('Class', axis=1)
Y = Dataframe['Class']

# Split into Training and test set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)
