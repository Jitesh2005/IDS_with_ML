# src/data_prep.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from utils import load_nslkdd, map_label_to_category

# Categorical features in NSL-KDD
CATEGORICAL = ['protocol_type', 'service', 'flag']


def prepare_data(train_file, test_file, save_dir='../models',
                 binary=True, apply_smote=False):

    # ---- Load raw dataset ----
    train = load_nslkdd(train_file)
    test = load_nslkdd(test_file)

    print("Loaded train:", train.shape)
    print("Loaded test:", test.shape)
    print("Label distribution:", train['label'].value_counts())

    # ---- Label mapping ----
    if binary:
        train['target'] = train['label'].apply(lambda x: 0 if x == 'normal' else 1)
        test['target'] = test['label'].apply(lambda x: 0 if x == 'normal' else 1)
    else:
        train['target'] = train['label'].apply(map_label_to_category)
        test['target'] = test['label'].apply(map_label_to_category)

    # ---- Separate X,y ----
    X_train = train.drop(columns=['label', 'target'])
    y_train = train['target'].values

    X_test = test.drop(columns=['label', 'target'])
    y_test = test['target'].values

    # ---- One-hot encode categorical features ----
    ohe = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    ohe.fit(X_train[CATEGORICAL])

    X_train_cat = pd.DataFrame(
        ohe.transform(X_train[CATEGORICAL]),
        index=X_train.index,
        columns=ohe.get_feature_names_out(CATEGORICAL)
    )

    X_test_cat = pd.DataFrame(
        ohe.transform(X_test[CATEGORICAL]),
        index=X_test.index,
        columns=ohe.get_feature_names_out(CATEGORICAL)
    )

    # ---- Numeric features ----
    num_cols = [c for c in X_train.columns if c not in CATEGORICAL]

    X_train_num = X_train[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_num = X_test[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # ---- Merge numeric + encoded ----
    X_train_final = pd.concat(
        [X_train_num.reset_index(drop=True), X_train_cat.reset_index(drop=True)],
        axis=1
    )

    X_test_final = pd.concat(
        [X_test_num.reset_index(drop=True), X_test_cat.reset_index(drop=True)],
        axis=1
    )

    # ---- Scale features ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_test_scaled = scaler.transform(X_test_final)

    # ---- Safe SMOTE ----
    if apply_smote and binary:
        if len(set(y_train)) > 1:
            print("Applying SMOTE...")
            # Use None for random_state to get different sampling each run
            sm = SMOTE(random_state=None)
            X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
        else:
            print("⚠️ WARNING: SMOTE skipped — only one class detected in training labels!")

    # ---- Save encoder + scaler ----
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(ohe, os.path.join(save_dir, "ohe.joblib"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    print("Preprocessing complete.")
    print("Final train shape:", X_train_scaled.shape)
    print("Final test shape:", X_test_scaled.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Debugging run
if __name__ == "__main__":
    train_file = "../data/KDDTrain+.txt"
    test_file = "../data/KDDTest+.txt"

    X_train, X_test, y_train, y_test = prepare_data(
        train_file, test_file,
        save_dir="../models",
        binary=True,
        apply_smote=True
    )
