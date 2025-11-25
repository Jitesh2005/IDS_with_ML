# Network Intrusion Detection System (NIDS)

A machine learning-based Network Intrusion Detection System using the NSL-KDD dataset. This project implements both supervised (XGBoost) and unsupervised (Autoencoder) approaches for detecting network attacks.

## 🎯 Project Overview

This IDS system uses two complementary approaches:
- **XGBoost Classifier**: Supervised learning for binary classification (normal vs attack)
- **Autoencoder**: Unsupervised anomaly detection based on reconstruction error

## 📊 Dataset

**NSL-KDD Dataset** - An improved version of the KDD Cup 1999 dataset
- Training samples: KDDTrain+.txt
- Testing samples: KDDTest+.txt
- 41 features including protocol type, service, flag, and various network statistics
- Binary classification: Normal vs Attack traffic

## 🚀 Features

- Comprehensive data preprocessing pipeline
- One-hot encoding for categorical features
- Standard scaling for numerical features
- SMOTE for handling class imbalance
- XGBoost classifier with optimized hyperparameters
- Autoencoder-based anomaly detection
- Performance evaluation with confusion matrices
- Feature importance visualization

## 📁 Project Structure

```
AI-ta2/
├── data/
│   ├── KDDTrain+.txt          # Training dataset
│   └── KDDTest+.txt           # Testing dataset
├── src/
│   ├── utils.py               # Helper functions and data loading
│   ├── data_prep.py           # Data preprocessing pipeline
│   ├── train_xgb.py           # XGBoost model training
│   ├── train_autoencoder.py   # Autoencoder training
│   └── evaluate.py            # Model evaluation
├── models/                     # Saved models and preprocessors
│   ├── xgb_model.joblib
│   ├── ae_model.h5
│   ├── scaler.joblib
│   └── ohe.joblib
├── plots/                      # Visualization outputs
│   ├── confusion_matrix_xgb.png
│   └── feature_importance_xgb.png
├── requirements.txt            # Python dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/network-ids.git
cd network-ids
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📥 Dataset Setup

Download the NSL-KDD dataset and place the files in the `data/` directory:
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

Required files:
- `KDDTrain+.txt`
- `KDDTest+.txt`

## 🎓 Usage

### 1. Train XGBoost Model
```bash
python src/train_xgb.py
```
This will:
- Preprocess the data
- Apply SMOTE for class balancing
- Train the XGBoost classifier
- Save the model and generate visualizations

### 2. Train Autoencoder
```bash
python src/train_autoencoder.py
```
This will:
- Train the autoencoder on normal traffic only
- Save the trained model

### 3. Evaluate Models
```bash
python src/evaluate.py
```
This will:
- Load both trained models
- Evaluate on test data
- Display confusion matrices and classification reports

## 📈 Model Performance

### XGBoost Classifier
- **Architecture**: Gradient boosting with 250 estimators
- **Max Depth**: 12
- **Learning Rate**: 0.15
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

### Autoencoder
- **Architecture**: 
  - Encoder: 128 → 64 → 32 neurons
  - Decoder: 32 → 64 → 128 neurons
- **Loss Function**: Mean Squared Error (MSE)
- **Anomaly Detection**: Threshold-based on reconstruction error

## 🔧 Configuration

### XGBoost Hyperparameters
Edit `src/train_xgb.py`:
```python
model = XGBClassifier(
    n_estimators=250,
    max_depth=12,
    learning_rate=0.15,
    subsample=0.75,
    colsample_bytree=0.75,
    random_state=None  # For reproducibility, set to an integer
)
```

### Autoencoder Parameters
Edit `src/train_autoencoder.py`:
```python
ae.fit(X_train_norm, X_train_norm, 
       epochs=30, 
       batch_size=256, 
       validation_split=0.1)
```

## 📊 Outputs

### Saved Models
- `models/xgb_model.joblib` - Trained XGBoost classifier
- `models/ae_model.h5` - Trained autoencoder
- `models/scaler.joblib` - Feature scaler
- `models/ohe.joblib` - One-hot encoder

### Visualizations
- `plots/confusion_matrix_xgb.png` - Confusion matrix heatmap
- `plots/feature_importance_xgb.png` - Top 20 important features

## 🧪 Testing

Run individual components:
```bash
# Test data preprocessing only
python src/data_prep.py

# Test utilities
python -c "from src.utils import load_nslkdd; print(load_nslkdd('data/KDDTrain+.txt').shape)"
```


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NSL-KDD Dataset creators
- XGBoost and TensorFlow/Keras communities
- scikit-learn and imbalanced-learn libraries
  

## 🔮 Future Improvements

- [ ] Add multi-class classification (DoS, Probe, R2L, U2R)
- [ ] Implement ensemble methods
- [ ] Add real-time detection capability
- [ ] Create web dashboard for monitoring
- [ ] Add more deep learning models (LSTM, CNN)
- [ ] Implement cross-validation
- [ ] Add hyperparameter tuning with GridSearch/RandomSearch

