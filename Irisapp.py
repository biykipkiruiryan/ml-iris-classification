# Importing necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load Iris dataset from sklearn
iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Species'] = iris.target

# Title and description of the app
st.title("Iris Flower Classification App")
st.write("""
This app classifies Iris flowers into three species: **Setosa**, **Versicolor**, or **Virginica**. 
You can adjust the features using the sliders and the decision tree classifier will predict the species.
""")
# Tab layout for displaying the corresponding species image based on prediction
tab1, tab2, tab3 = st.tabs(["Versicolor", "Setosa", "Virginica"])
with tab1:
    st.header("Versicolor")
    st.image("https://daylily-phlox.eu/wp-content/uploads/2016/08/Iris-versicolor-1.jpg", width=180)

with tab2:
    st.header("Setosa")
    st.image("https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg", width=180)

with tab3:
    st.header("Virginica")
    st.image("https://wiki.irises.org/pub//Spec/SpecVirginica/ivirginicagiantblue01.jpg", width=180)

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

# Displaying the result
st.subheader("Prediction:")
st.write(f"The predicted species is: **{predicted_species}**")
# Display the corresponding species image right after the prediction
if predicted_species == 'Versicolor':
    st.image("https://daylily-phlox.eu/wp-content/uploads/2016/08/Iris-versicolor-1.jpg", width=180)
elif predicted_species == 'Setosa':
    st.image("https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg", width=180)
elif predicted_species == 'Virginica':
    st.image("https://wiki.irises.org/pub//Spec/SpecVirginica/ivirginicagiantblue01.jpg", width=180)

# Test the model on test data and show accuracy
prediction_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, prediction_dt) * 100

st.subheader("Model Accuracy:")
st.write(f"Accuracy of the Decision Tree classifier on test data: {accuracy_dt:.2f}%")

# Plot the decision tree
st.subheader("Decision Tree Visualization")
plt.figure(figsize=(12, 8))
tree.plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
st.pyplot(plt)
