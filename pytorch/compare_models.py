#!/usr/bin/env python
# coding: utf-8
"""
Benchmark Multi-class Respiratory Disease Classification
So sánh các thuật toán Machine Learning khác nhau trên đặc trưng FRILL.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

SOUNDDR_DIR = os.path.join(os.path.dirname(__file__), '..', 'Sound-Dr')
DATA_DIR    = os.path.join(SOUNDDR_DIR, 'sounddr_data')
CSV_PATH    = os.path.join(DATA_DIR, 'data.csv')
CACHE_FRILL = os.path.join(DATA_DIR, 'output', 'FRILL.pickle')

FOLD_NUM = 5
SEED     = 2022

RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']

def get_respiratory_label(row) -> int:
    cov  = str(row.get('cov19_status_choice', 'never')).strip().lower()
    cond = str(row.get('medical_condition_choice', "['No']")).lower()
    if cov in ('last14', 'over14'): return 4
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')): return 3
    if 'copd' in cond: return 2
    if 'asthma' in cond: return 1
    return 0

# 1. Chuẩn bị Data
df = pd.read_csv(CSV_PATH)
if 'error' in df.columns:
    df = df[df['error'] == 0].reset_index(drop=True)

df['label_respiratory'] = df.apply(get_respiratory_label, axis=1)

le = LabelEncoder()
y = le.fit_transform(df['label_respiratory'].values)
num_active_classes = len(le.classes_)

with open(CACHE_FRILL, 'rb') as f:
    cache = pickle.load(f)
X = cache.get('X_trill_features', cache.get('X_frill'))

Fold = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
folds = np.zeros(len(df), dtype=int)
for n, (_, val_idx) in enumerate(Fold.split(X, y)):
    folds[val_idx] = n

# Calculate generic class weights (for models that support class_weight='balanced')
# For XGBoost we calculate sample_weights
class_counts = np.bincount(y, minlength=num_active_classes).astype(float)
total = class_counts.sum()
sample_weight = np.array([total / (num_active_classes * class_counts[yi]) for yi in y])

# 2. Khai báo các thiết lập Model
models = {
    "XGBoost": lambda: xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_active_classes,
        eval_metric='mlogloss',
        max_depth=4, learning_rate=0.1, n_estimators=50,
        subsample=0.8, colsample_bytree=0.8, seed=SEED, nthread=-1, verbosity=0
    ),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=50, class_weight='balanced', random_state=SEED, n_jobs=-1
    ),
    "SVM (RBF Kernel)": lambda: SVC(
        probability=True, class_weight='balanced', random_state=SEED
    ),
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=100, class_weight='balanced', random_state=SEED, n_jobs=-1
    )
}

results = []

print("=" * 60)
print("  BENCHMARK: COMPARISON OF ML ALGORITHMS ON FRILL FEATURES")
print("=" * 60)

for model_name, model_fn in models.items():
    print(f"\n[INFO] Đang đánh giá: {model_name}...")
    all_targets = []
    all_preds   = []

    for fold in range(FOLD_NUM):
        train_idx = folds != fold
        val_idx   = folds == fold
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        
        model = model_fn()
        
        # XGBoost handles sample_weight via fit
        if model_name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
            preds_val = np.argmax(model.predict_proba(X_val), axis=1)
        else:
            # sklearn models handle class imbalance via class_weight='balanced'
            model.fit(X_train, y_train)
            preds_val = model.predict(X_val)
            
        all_targets.extend(y_val.tolist())
        all_preds.extend(preds_val.tolist())
        
    acc = accuracy_score(all_targets, all_preds)
    f1  = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    results.append({
        'Algorithm': model_name,
        'Accuracy': f"{acc * 100:.2f}%",
        'Macro F1': f"{f1 * 100:.2f}%"
    })
    print(f"       → Accuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%")

print("\n" + "=" * 60)
print("  KẾT QUẢ TỔNG HỢP")
print("=" * 60)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
print("=" * 60)
