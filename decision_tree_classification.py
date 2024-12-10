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







