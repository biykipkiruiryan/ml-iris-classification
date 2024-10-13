import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset from sklearn
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Species'] = iris.target

st.title("Iris Flower Classification App")
st.write("""
This app classifies Iris flowers into three species: **Setosa**, **Versicolor**, or **Virginica**. 
You can adjust the features using the sliders and the decision tree classifier will predict the species.
""")

# Main page sliders for user input
sepal_length = st.slider('Sepal Length (cm)', float(iris_df['sepal length (cm)'].min()), float(iris_df['sepal length (cm)'].max()), float(iris_df['sepal length (cm)'].mean()))
sepal_width = st.slider('Sepal Width (cm)', float(iris_df['sepal width (cm)'].min()), float(iris_df['sepal width (cm)'].max()), float(iris_df['sepal width (cm)'].mean()))
petal_length = st.slider('Petal Length (cm)', float(iris_df['petal length (cm)'].min()), float(iris_df['petal length (cm)'].max()), float(iris_df['petal length (cm)'].mean()))
petal_width = st.slider('Petal Width (cm)', float(iris_df['petal width (cm)'].min()), float(iris_df['petal width (cm)'].max()), float(iris_df['petal width (cm)'].mean()))

# Collecting input from the sliders
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Display user input
st.subheader("Your Input:")
st.table(input_data)

# Splitting dataset into train and test sets
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Train Decision Tree classifier
dt = DecisionTreeClassifier(random_state=100)
dt.fit(X_train, y_train)

# Predict based on user input
prediction = dt.predict(input_data)

# Mapping predicted value to the class label
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_species = species_map[prediction[0]]

# Display the result
st.subheader("Prediction:")
st.write(f"The predicted species is: **{predicted_species}**")

# Test the model and show accuracy
prediction_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, prediction_dt) * 100

st.subheader("Model Accuracy:")
st.write(f"Accuracy of the Decision Tree classifier on test data: {accuracy_dt:.2f}%")
