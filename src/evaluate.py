# src/evaluate.py
import os, joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from tensorflow.keras.models import load_model
from data_prep import prepare_data
from utils import load_nslkdd

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def evaluate():
    train_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'KDDTrain+.txt')
    test_file  = os.path.join(os.path.dirname(__file__), '..', 'data', 'KDDTest+.txt')
    X_train, X_test, y_train, y_test = prepare_data(train_file, test_file, save_dir=MODEL_DIR, binary=True, apply_smote=False)
    
    # Add some randomization to test set if needed
    import numpy as np
    np.random.seed(None)  # Use different seed each time

    # XG-Boost
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    preds = xgb.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Autoencoder
    ae = load_model(os.path.join(MODEL_DIR, 'ae_model.h5'))
    recon = ae.predict(X_test)
    mse = np.mean(np.square(X_test - recon), axis=1)

    # set threshold from training normal set: estimate from train normal
    df_train = load_nslkdd(train_file)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    ohe = joblib.load(os.path.join(MODEL_DIR, 'ohe.joblib'))

    # compute threshold using a sample of training normals (this is simplified)
    X_train_all = X_train
    df_train = load_nslkdd(train_file)
    normal_mask = (df_train['label'] == 'normal').values
    normal_train = X_train_all[normal_mask]
    
    # Add some randomization to threshold calculation
    sample_size = min(1000, len(normal_train))  # Use subset for variation
    indices = np.random.choice(len(normal_train), sample_size, replace=False)
    normal_sample = normal_train[indices]
    
    recon_norm = ae.predict(normal_sample)
    mse_norm = np.mean(np.square(normal_sample - recon_norm), axis=1)
    
    # Vary threshold percentile slightly
    percentile = np.random.uniform(90, 98)  # Random between 90-98th percentile
    thresh = np.percentile(mse_norm, percentile)
    print(f"AE threshold ({percentile:.1f}th pct):", thresh)

    ae_preds = (mse > thresh).astype(int)  # 1 = anomaly
    print("Autoencoder classification report (anomaly=1):")
    print(classification_report(y_test, ae_preds))
    print("AE confusion matrix:\n", confusion_matrix(y_test, ae_preds))

if __name__ == "__main__":
    evaluate()
