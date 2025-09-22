# train_model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load sample dataset (Iris)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train decision tree
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "decision_tree_model.pkl")

print("Model trained and saved as decision_tree_model.pkl")