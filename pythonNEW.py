# Import necessary libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Load Training Values
train_values = pd.read_csv('train_values.csv', index_col='building_id')

# Load Training Labels
train_labels = pd.read_csv('train_labels.csv', index_col='building_id')

# Perform one-hot encoding on categorical variables
train_values_encoded = pd.get_dummies(train_values)

# Create the Random Forest classifier
model = RandomForestClassifier(random_state=2018)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_values_encoded, train_labels.values.ravel(), test_size=0.2, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = model.predict(X_val)

# Calculate the F1 score on the validation set
f1_micro_train = f1_score(y_val, val_predictions, average='micro')

print("Micro-averaged F1 Score on Training Data:", f1_micro_train)

# Load Test Values
test_values = pd.read_csv('test_values.csv', index_col='building_id')

# Perform one-hot encoding on categorical variables for the test dataset
test_values_encoded = pd.get_dummies(test_values)

# Make predictions using the trained model
predictions = model.predict(test_values_encoded)

# Load the submission format
submission_format = pd.read_csv('submission_format.csv', index_col='building_id')

# Create a DataFrame for our predictions with the correct column names and index
my_submission = pd.DataFrame(data=predictions, columns=submission_format.columns, index=submission_format.index)

# Save the submission DataFrame to a CSV file
my_submission.to_csv('submission.csv')

# Display the head of the saved file
my_submission.head()