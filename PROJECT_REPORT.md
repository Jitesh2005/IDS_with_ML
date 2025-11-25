# Network Intrusion Detection System (IDS) - Project Report

## Executive Summary

This project implements a hybrid machine learning-based Intrusion Detection System (IDS) using the NSL-KDD dataset. The system combines two complementary approaches: supervised learning (XGBoost) and unsupervised learning (Autoencoder) to detect network intrusions and anomalies.

**Key Highlights:**
- Dataset: NSL-KDD (improved version of KDD Cup 99)
- Primary Model: XGBoost Classifier
- Secondary Model: Deep Autoencoder
- Binary Classification: Normal vs Attack
- Feature Engineering: One-Hot Encoding + Standard Scaling
- Class Balancing: SMOTE (Synthetic Minority Over-sampling)

---

## Project Structure

```
AI-ta2/
├── data/
│   ├── KDDTrain+.txt          # Training dataset
│   └── KDDTest+.txt           # Testing dataset
├── models/
│   ├── xgb_model.joblib       # Trained XGBoost model
│   ├── ae_model.h5            # Trained Autoencoder (if exists)
│   ├── scaler.joblib          # StandardScaler for normalization
│   └── ohe.joblib             # OneHotEncoder for categorical features
├── plots/
│   ├── confusion_matrix_xgb.png
│   └── feature_importance_xgb.png
├── src/
│   ├── utils.py               # Data loading and label mapping utilities
│   ├── data_prep.py           # Data preprocessing pipeline
│   ├── train_xgb.py           # XGBoost training script
│   ├── train_autoencoder.py   # Autoencoder training script
│   └── evaluate.py            # Model evaluation script
├── requirements.txt           # Python dependencies
└── venv/                      # Virtual environment
```

---

## Dataset: NSL-KDD

### Overview
NSL-KDD is an improved version of the KDD Cup 99 dataset, addressing issues like redundant records and class imbalance. It contains network traffic features extracted from TCP/IP connections.

### Features (41 total)
1. **Basic Features (9)**: Duration, protocol type, service, flag, bytes transferred, etc.
2. **Content Features (13)**: Failed logins, root shell access, file creations, etc.
3. **Traffic Features (9)**: Connection counts, error rates, service rates
4. **Host-based Features (10)**: Destination host statistics

### Categorical Features
- `protocol_type`: tcp, udp, icmp
- `service`: http, ftp, smtp, etc. (70+ services)
- `flag`: Connection status flags (SF, REJ, RSTO, etc.)

### Attack Categories
- **DoS**: Denial of Service (neptune, smurf, pod, teardrop, etc.)
- **Probe**: Surveillance and probing (portsweep, ipsweep, nmap, satan)
- **R2L**: Remote to Local attacks (guess_passwd, ftp_write, imap, phf)
- **U2R**: User to Root attacks (buffer_overflow, rootkit, loadmodule, perl)

---

## Data Preprocessing Pipeline

### 1. Data Loading (`utils.py`)
- Loads raw NSL-KDD files (42 columns including difficulty score)
- Removes difficulty score column
- Assigns proper column names
- Cleans label strings

### 2. Label Encoding
**Binary Mode** (Current Implementation):
- Normal → 0
- All Attacks → 1

**Multi-class Mode** (Available):
- Maps attacks to 5 categories: normal, dos, probe, r2l, u2r

### 3. Feature Engineering
**Categorical Encoding:**
- OneHotEncoder for protocol_type, service, flag
- Handles unknown categories gracefully
- Creates ~120+ features after encoding

**Numerical Processing:**
- Converts all numeric columns to float
- Fills missing values with 0
- Preserves 38 original numeric features

### 4. Feature Scaling
- StandardScaler (zero mean, unit variance)
- Fitted on training data only
- Applied to both train and test sets

### 5. Class Balancing (SMOTE)
- Applied only during XGBoost training
- Generates synthetic minority class samples
- Random state removed for variation between runs
- Balances normal vs attack classes

**Final Feature Count:** ~158 features (38 numeric + 120 encoded categorical)

---

## Model 1: XGBoost Classifier

### Architecture
```python
XGBClassifier(
    n_estimators=250,        # 250 decision trees
    max_depth=12,            # Maximum tree depth
    learning_rate=0.15,      # Step size shrinkage
    subsample=0.75,          # 75% row sampling
    colsample_bytree=0.75,   # 75% feature sampling
    eval_metric="logloss",   # Binary cross-entropy
    tree_method="hist",      # Fast histogram-based algorithm
    n_jobs=-1,               # Use all CPU cores
    random_state=None        # Different results each run
)
```

### Training Process
1. Load and preprocess data with SMOTE
2. Train on balanced dataset
3. Evaluate on original test set
4. Save model and generate visualizations

### Hyperparameters Rationale
- **High n_estimators (250)**: Complex patterns in network traffic
- **Deep trees (12)**: Capture intricate attack signatures
- **Moderate learning rate (0.15)**: Balance speed and accuracy
- **Subsampling (0.75)**: Prevent overfitting, improve generalization
- **Histogram method**: Faster training on large datasets

### Outputs
- Trained model: `models/xgb_model.joblib`
- Confusion matrix visualization
- Top 20 feature importance plot
- Classification report (precision, recall, F1-score)

---

## Model 2: Deep Autoencoder

### Architecture
```
Input Layer:     158 features
Encoder:         Dense(128, relu) → Dense(64, relu) → Dense(32, relu)
Decoder:         Dense(64, relu) → Dense(128, relu) → Dense(158, linear)
Loss Function:   Mean Squared Error (MSE)
Optimizer:       Adam
```

### Training Strategy
1. **Train on normal traffic only** (unsupervised anomaly detection)
2. Learn to reconstruct normal network behavior
3. High reconstruction error → Anomaly/Attack

### Anomaly Detection Process
1. Calculate reconstruction error (MSE) for each sample
2. Compute threshold from normal training samples
3. Dynamic threshold: 90-98th percentile (randomized)
4. Samples exceeding threshold → Classified as attacks

### Key Features
- Unsupervised learning approach
- Complementary to XGBoost
- Detects novel/unknown attacks
- No attack samples needed for training

### Outputs
- Trained model: `models/ae_model.h5`
- Dynamic threshold calculation
- Anomaly predictions based on reconstruction error

---

## Evaluation Metrics

### Classification Metrics
1. **Accuracy**: Overall correct predictions
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix
```
                Predicted
              Normal  Attack
Actual Normal   TN      FP
       Attack   FN      TP
```

- **True Negative (TN)**: Normal correctly identified
- **False Positive (FP)**: Normal misclassified as attack
- **False Negative (FN)**: Attack missed (critical!)
- **True Positive (TP)**: Attack correctly detected

### Performance Considerations
- **False Negatives** are critical in IDS (missed attacks)
- **False Positives** cause alert fatigue
- Balance between detection rate and false alarm rate

---

## Key Implementation Features

### 1. Randomization for Variation
Recent updates ensure different results on each run:
- XGBoost: `random_state=None`
- SMOTE: `random_state=None`
- Autoencoder: Shuffling enabled, dynamic threshold
- Evaluation: Random sampling for threshold calculation

### 2. Modular Design
- Separate scripts for each task
- Reusable preprocessing pipeline
- Easy to extend with new models
- Clear separation of concerns

### 3. Artifact Management
- Models saved as joblib/h5 files
- Preprocessors (scaler, encoder) persisted
- Visualizations automatically generated
- Reproducible pipeline

### 4. Error Handling
- Graceful handling of missing values
- Unknown category support in encoding
- SMOTE safety checks for single-class scenarios
- Robust file path management

---

## Dependencies

### Core Libraries
- **numpy** (≥1.21): Numerical computations
- **pandas** (≥1.3): Data manipulation
- **scikit-learn** (≥1.0): ML utilities, preprocessing, metrics

### Machine Learning
- **xgboost** (≥1.6): Gradient boosting classifier
- **tensorflow** (≥2.6): Deep learning framework
- **keras** (≥2.6): Neural network API

### Data Processing
- **imbalanced-learn** (≥0.9): SMOTE implementation
- **joblib** (≥1.1): Model serialization

### Visualization
- **matplotlib** (≥3.4): Plotting library
- **seaborn** (≥0.11): Statistical visualizations

---

## Usage Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train XGBoost Model
```bash
python src/train_xgb.py
```
**Outputs:**
- `models/xgb_model.joblib`
- `plots/confusion_matrix_xgb.png`
- `plots/feature_importance_xgb.png`
- Console: Accuracy, classification report

### 3. Train Autoencoder
```bash
python src/train_autoencoder.py
```
**Outputs:**
- `models/ae_model.h5`
- Console: Training progress (30 epochs)

### 4. Evaluate Both Models
```bash
python src/evaluate.py
```
**Outputs:**
- XGBoost performance metrics
- Autoencoder anomaly detection results
- Confusion matrices for both models
- Dynamic threshold information

---

## Strengths of the Approach

### 1. Hybrid Detection
- **XGBoost**: Excellent for known attack patterns
- **Autoencoder**: Detects novel/zero-day attacks
- Complementary strengths

### 2. Robust Preprocessing
- Handles mixed data types (numeric + categorical)
- Proper scaling and encoding
- Class balancing with SMOTE

### 3. Production-Ready
- Modular, maintainable code
- Saved models for deployment
- Clear evaluation metrics
- Visualization for interpretability

### 4. Flexibility
- Easy to switch between binary/multi-class
- Configurable hyperparameters
- Extensible architecture

---

## Potential Improvements

### 1. Model Enhancements
- Ensemble methods (voting, stacking)
- Deep learning classifiers (LSTM, CNN)
- Feature selection/dimensionality reduction
- Hyperparameter tuning (GridSearch, Optuna)

### 2. Evaluation
- Cross-validation for robustness
- ROC curves and AUC scores
- Per-attack-type performance analysis
- Real-time performance metrics

### 3. Deployment
- REST API for predictions
- Real-time streaming detection
- Model monitoring and retraining
- Alert system integration

### 4. Data
- Feature engineering (temporal patterns)
- Additional datasets for validation
- Adversarial attack testing
- Concept drift handling

---

## Conclusion

This project demonstrates a comprehensive approach to network intrusion detection using modern machine learning techniques. The combination of XGBoost for supervised learning and Autoencoder for anomaly detection provides robust coverage for both known and unknown threats.

The modular architecture, proper preprocessing pipeline, and evaluation framework make this system suitable for further development and potential deployment in real-world network security scenarios.

**Project Status:** ✅ Functional and ready for evaluation
**Next Steps:** Train models, evaluate performance, iterate on improvements

---

## References

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- XGBoost Documentation: https://xgboost.readthedocs.io/
- TensorFlow/Keras: https://www.tensorflow.org/
- Scikit-learn: https://scikit-learn.org/

---

*Report Generated: November 23, 2025*
*Project: Network Intrusion Detection System*
*Author: AI-ta2 Development Team*
