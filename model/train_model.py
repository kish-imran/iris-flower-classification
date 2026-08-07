import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("=" * 50)
print("IRIS FLOWER CLASSIFICATION SYSTEM")
print("=" * 50)

# Load Dataset
try:
    df = pd.read_csv("dataset/iris.csv")
except FileNotFoundError:
    print("\nError: dataset/iris.csv not found!")
    exit()

print("\nDataset Loaded Successfully!\n")

print(df.head())

# Features and Target
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# Create model folder
os.makedirs("model", exist_ok=True)

# Save Model
joblib.dump(model, "model/iris_model.pkl")

print("\nModel Saved Successfully!")
print("Location: model/iris_model.pkl")