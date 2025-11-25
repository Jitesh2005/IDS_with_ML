# src/train_xgb.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_prep import prepare_data

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def train_and_save_xgb():

    print("Loading & preprocessing dataset...")
    train_file = os.path.join(os.path.dirname(__file__), "..", "data", "KDDTrain+.txt")
    test_file  = os.path.join(os.path.dirname(__file__), "..", "data", "KDDTest+.txt")

    X_train, X_test, y_train, y_test = prepare_data(
        train_file, test_file,
        save_dir=MODEL_DIR,
        binary=True,
        apply_smote=True
    )

    print("Training XGBoost Classifier...")

    model = XGBClassifier(
        n_estimators=250,
        max_depth=12,
        learning_rate=0.15,
        subsample=0.75,
        colsample_bytree=0.75,
        eval_metric="logloss",
        tree_method="hist",   # fast for CPU
        n_jobs=-1,
        random_state=None  # This will use different random seed each run
    )

    model.fit(X_train, y_train)

    # ---- Predictions ----
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nXGBoost Accuracy: {acc}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    print("\nConfusion Matrix:")
    print(cm)

    # ---- Save Model ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    print("\nSaved XGBoost model successfully.")

    # ---- Plots ----
    # Confusion matrix plot
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix - XGBoost IDS")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix_xgb.png"))
    plt.close()

    # Feature importance
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    idx = np.argsort(importance)[-20:]  # top 20 features
    plt.barh(range(20), importance[idx], color="green")
    plt.yticks(range(20), idx)
    plt.title("Top 20 Feature Importances - XGBoost IDS")
    plt.savefig(os.path.join(PLOT_DIR, "feature_importance_xgb.png"))
    plt.close()

    print("\nPlots saved inside the /plots folder.")


if __name__ == "__main__":
    train_and_save_xgb()
