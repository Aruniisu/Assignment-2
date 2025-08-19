# model_training.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pandas as pd

def train_and_save_model():
    """Train and save an Iris classification model."""
    try:
        # Load dataset
        print("Loading Iris dataset...")
        iris = load_iris()
        X, y = iris.data, iris.target
        feature_names = iris.feature_names
        target_names = iris.target_names

        # Split data
        print("Splitting dataset into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training Random Forest classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        print("\nModel Evaluation:")
        print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
        print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
        
        # Detailed classification report
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Save model
        print("\nSaving model...")
        model_data = {
            "model": model,
            "feature_names": feature_names,
            "target_names": target_names
        }
        joblib.dump(model_data, "model.pkl")
        print("Model successfully saved as model.pkl")

        return model

    except Exception as e:
        print(f"Error occurred during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_save_model()