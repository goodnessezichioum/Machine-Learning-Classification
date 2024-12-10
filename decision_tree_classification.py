import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Based on prior knowledge, the dataset might not have column names; we'll define them.
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load the data
cleveland_data = pd.read_csv("processed.cleveland.data", names=columns, na_values="?", header=None)

# Display the first few rows and general information
cleveland_data.info()

# Handle Missing Values ( number of missing values is negligible so drop missing data)
data = cleveland_data.dropna()

missing_values = cleveland_data.isnull().sum()
print(missing_values)

missing_val = data.isnull().sum()
print(missing_val)

# Convert target to binary
data.loc[:, 'target'] = (data['target'] > 0).astype(int)

from sklearn.model_selection import train_test_split

# Features and target
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Initialize a decision tree classifier (no pruning)
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Unpruned Decision Tree Accuracy: {accuracy:.4f}")

import matplotlib.pyplot as plt

# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True, rounded=True)
plt.title("Unpruned Decision Tree")
#plt.show()

# Initialize a decision tree classifier with max depth to prevent overfitting
dt_pruned_model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the pruned decision tree
dt_pruned_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_pruned = dt_pruned_model.predict(X_test)

# Evaluate the model accuracy
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print(f"Pruned Decision Tree Accuracy: {accuracy_pruned:.4f}")







