#!/usr/bin/env python
# coding: utf-8
"""
Chiến lược 1 – SMOTE + Augmentation + XGBoost Benchmark
So sánh 4 phương án:
  1. Baseline:    FRILL gốc, không xử lý mất cân bằng
  2. SMOTE only:  FRILL gốc + SMOTE oversampling
  3. Aug only:    FRILL gốc + MFCC-augmented (Chiến lược 2)
  4. Aug + SMOTE: Kết hợp cả hai  ← kỳ vọng tốt nhất

Run:
  source .venv/bin/activate && python pytorch/smote_benchmark.py
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb

SOUNDDR     = os.path.join(os.path.dirname(__file__), '..', 'Sound-Dr', 'sounddr_data')
CSV_AUG     = os.path.join(SOUNDDR, 'data_augmented.csv')
FRILL_CACHE = os.path.join(SOUNDDR, 'output', 'FRILL.pickle')
MFCC_AUG    = os.path.join(SOUNDDR, 'output', 'MFCC_augmented.pickle')
SEED, FOLDS = 2022, 5
RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']

# ─── Hàm gán nhãn ────────────────────────────────────────────────────────────
def get_label(row) -> int:
    cov  = str(row.get('cov19_status_choice', 'never')).lower()
    cond = str(row.get('medical_condition_choice', 'no')).lower()
    if cov in ('last14', 'over14'): return 4
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')): return 3
    if 'copd' in cond: return 2
    if 'asthma' in cond: return 1
    return 0

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_AUG)
df['label_respiratory'] = df.apply(get_label, axis=1)

# FRILL features (original 1310 samples, dim=4096)
with open(FRILL_CACHE, 'rb') as f:
    frill = pickle.load(f)
X_orig = frill.get('X_trill_features', frill.get('X_frill'))  # (1310, 4096)
y_orig = df[df['augmented'].fillna(False) == False]['label_respiratory'].values

# MFCC features for augmented samples (dim=256, padded to 4096)
with open(MFCC_AUG, 'rb') as f:
    mfcc = pickle.load(f)
X_mfcc = mfcc['X_mfcc_aug']  # (45, 256)
frill_dim = X_orig.shape[1]
X_mfcc_padded = np.zeros((len(X_mfcc), frill_dim), dtype=np.float32)
X_mfcc_padded[:, :X_mfcc.shape[1]] = X_mfcc
y_aug = df[df['augmented'].fillna(False) == True]['label_respiratory'].values

# Combined (original + augmented)
X_combined = np.vstack([X_orig, X_mfcc_padded])
y_combined  = np.concatenate([y_orig, y_aug])

print("=" * 60)
print("  SMOTE BENCHMARK – FRILL + Augmentation + SMOTE")
print("=" * 60)
print(f"\n[INFO] Dataset size: orig={len(X_orig)}, aug={len(X_mfcc)}, combined={len(X_combined)}")

# ─── Helper: run XGBoost cross-validation ─────────────────────────────────────
def run_cv(X, y, use_smote=False, label=""):
    le  = LabelEncoder()
    y_e = le.fit_transform(y)
    n   = len(le.classes_)
    active = [RESP_CLASSES[i] for i in le.classes_]

    Fold  = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(y_e), dtype=int)
    for i, (_, vi) in enumerate(Fold.split(X, y_e)):
        folds[vi] = i

    all_t, all_p = [], []
    for fold in range(FOLDS):
        ti, vi = folds != fold, folds == fold
        X_tr, y_tr = X[ti], y_e[ti]
        X_val, y_val = X[vi], y_e[vi]

        if use_smote:
            k = max(1, min(2, min(np.bincount(y_tr)) - 1))   # k_neighbors an toàn
            try:
                sm = SMOTE(k_neighbors=k, random_state=SEED)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception as e:
                pass  # Bỏ qua nếu class quá nhỏ

        counts = np.bincount(y_tr, minlength=n).astype(float)
        sw = np.array([counts.sum() / (n * counts[yi]) for yi in y_tr])

        mdl = xgb.XGBClassifier(
            objective='multi:softprob', num_class=n, eval_metric='mlogloss',
            max_depth=4, learning_rate=0.1, n_estimators=100,
            subsample=0.8, colsample_bytree=0.8, seed=SEED, nthread=-1, verbosity=0
        )
        mdl.fit(X_tr, y_tr, sample_weight=sw)
        pv = np.argmax(mdl.predict_proba(X_val), axis=1)
        all_t.extend(y_val); all_p.extend(pv)

    acc = accuracy_score(all_t, all_p)
    f1  = f1_score(all_t, all_p, average='macro', zero_division=0)
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Accuracy : {acc*100:.2f}%   |   Macro F1 : {f1*100:.2f}%")
    print(classification_report(all_t, all_p, target_names=active, zero_division=0))
    return acc, f1


# ─── Chạy 4 phương án ─────────────────────────────────────────────────────────
results = []

acc, f1 = run_cv(X_orig, y_orig, use_smote=False,
                 label="1. Baseline (FRILL gốc, không xử lý)")
results.append(("Baseline", f"{acc*100:.2f}%", f"{f1*100:.2f}%"))

acc, f1 = run_cv(X_orig, y_orig, use_smote=True,
                 label="2. SMOTE only (FRILL gốc + SMOTE)")
results.append(("SMOTE only", f"{acc*100:.2f}%", f"{f1*100:.2f}%"))

acc, f1 = run_cv(X_combined, y_combined, use_smote=False,
                 label="3. Aug only (FRILL + MFCC-augmented)")
results.append(("Aug only", f"{acc*100:.2f}%", f"{f1*100:.2f}%"))

acc, f1 = run_cv(X_combined, y_combined, use_smote=True,
                 label="4. Aug + SMOTE (tốt nhất kỳ vọng)")
results.append(("Aug + SMOTE", f"{acc*100:.2f}%", f"{f1*100:.2f}%"))

# ─── Bảng tổng hợp ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BẢNG SO SÁNH TỔNG HỢP")
print("=" * 60)
res_df = pd.DataFrame(results, columns=["Phương án", "Accuracy", "Macro F1"])
print(res_df.to_string(index=False))
print("=" * 60)
