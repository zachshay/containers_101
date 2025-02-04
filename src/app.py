from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# needed for path creation
import os

# Load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Create a Gaussian Classifier
clf = RandomForestClassifier()

# Train the model using the training sets
clf.fit(x_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(x_test)

# Model Accuracy, how often is the classifier correct?
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Create the 'models' directory if it doesn't exist
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Construct the full path to the model file
model_path = os.path.join(models_dir, "iris_model.pkl")

# Save the trained model
joblib.dump(clf, model_path)
print(f"Model saved to: {model_path}!")