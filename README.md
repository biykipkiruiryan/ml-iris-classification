# Preparing The Data:

First of all, we loaded the Iris dataset from the scikit-learn library.
We printed the dataset, feature names & labels (target names).
We split the data (features, labels) into 70% training, 30% testing.

# Decision Tree Algorithm:
We defined a decision tree classifier.
We trained the model using 'fit' function on the 70% data we put aside for training.
We tested the X_test data we put aside for testing, using 'predict' function & got predict_dt.
We checked the accuracy of our results against the y_test data we put aside - multiplied by 100 to get %, and got accuracy of approx. 95%.
We visualized the decision tree.