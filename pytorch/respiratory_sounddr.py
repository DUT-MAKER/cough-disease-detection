#!/usr/bin/env python
# coding: utf-8
"""
Track B – FRILL Embedding + XGBoost
Phân loại đa lớp bệnh hô hấp từ dữ liệu Sound-Dr.

Classes:
  0 – Healthy    (không triệu chứng / không bệnh lý)
  1 – Asthma     (hen suyễn)
  2 – COPD       (bệnh phổi tắc nghẽn mãn tính)
  3 – Pneumonia  (viêm phổi / bệnh phổi mức độ nặng)
  4 – COVID-19

Run:
  cd /home/nguyenhuynh/Documents/detect-cough-disease-with-CovNet
  uv run python pytorch/respiratory_sounddr.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, pickle

# Thêm Sound-Dr vào sys.path để import feature.py của Sound-Dr
SOUNDDR_DIR = os.path.join(os.path.dirname(__file__), '..', 'Sound-Dr')
sys.path.insert(0, os.path.abspath(SOUNDDR_DIR))

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, accuracy_score, f1_score
)
from tqdm import tqdm

# Từ Sound-Dr

# ─── Cấu hình ────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(SOUNDDR_DIR, 'sounddr_data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
CSV_PATH_AUG = os.path.join(DATA_DIR, 'data_augmented.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output_respiratory')
CACHE_FRILL = os.path.join(DATA_DIR, 'output', 'FRILL.pickle') # Sủ dụng cache cũ của Sound-Dr

FOLD_NUM = 5
SEED     = 2022

RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']
NUM_CLASSES  = len(RESP_CLASSES)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Hàm gán nhãn đa lớp ────────────────────────────────────────────────────
def get_respiratory_label(row) -> int:
    """
    Ưu tiên:  COVID (4) > Pneumonia/phổi nặng (3) > COPD (2) > Asthma (1) > Healthy (0)
    """
    cov  = str(row.get('cov19_status_choice', 'never')).strip().lower()
    cond = str(row.get('medical_condition_choice', "['No']")).lower()

    if cov in ('last14', 'over14'):
        return 4  # COVID

    # Bệnh phổi nặng / viêm phổi
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')):
        return 3

    # COPD
    if 'copd' in cond:
        return 2

    # Asthma + các bệnh tắc nghẽn khác
    if 'asthma' in cond:
        return 1

    return 0  # Healthy


# ─── Đọc dữ liệu ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  Respiratory Disease Classification – FRILL + XGBoost")
print("=" * 60)

CSV_IN_USE = CSV_PATH_AUG if os.path.isfile(CSV_PATH_AUG) else CSV_PATH
print(f"[INFO] Dùng CSV train: {CSV_IN_USE}")

df = pd.read_csv(CSV_IN_USE)
df['file_path'] = df['file_cough'] + '.wav'
df['label_respiratory'] = df.apply(get_respiratory_label, axis=1)

# Thống kê phân phối nhãn
print("\n[INFO] Phân phối nhãn:")
label_counts = df['label_respiratory'].value_counts().sort_index()
for idx, count in label_counts.items():
    print(f"  {idx} – {RESP_CLASSES[idx]:12s}: {count:4d} mẫu")
print(f"  Tổng: {len(df)} mẫu\n")

# Lọc bỏ các dòng lỗi (error == 1) nếu có cột error
if 'error' in df.columns:
    df = df[df['error'] == 0].reset_index(drop=True)
    print(f"[INFO] Sau khi lọc error=0: {len(df)} mẫu\n")

# ─── Trích xuất đặc trưng FRILL ───────────────────────────────────────────────
print("[INFO] Trích xuất FRILL embeddings...")
if os.path.exists(CACHE_FRILL):
    print(f"  → Dùng cache: {CACHE_FRILL}")
    with open(CACHE_FRILL, 'rb') as f:
        cache = pickle.load(f)
    if 'X_trill_features' in cache:
        X_frill = cache['X_trill_features']
    else:
        X_frill = cache['X_frill']
else:
    X_frill = []
    errors  = []
    from feature import make_nonsemantic_frill_nofrontend_feat
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = os.path.join(DATA_DIR, row['file_path'])
        try:
            feat = make_nonsemantic_frill_nofrontend_feat(audio_path)
            X_frill.append(feat)
        except Exception as e:
            X_frill.append(np.zeros(2048))   # fallback vector zero
            errors.append((idx, str(e)))

    X_frill = np.array(X_frill, dtype=np.float32)
    X_frill = np.nan_to_num(X_frill)
    X_frill = np.clip(X_frill, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    with open(CACHE_FRILL, 'wb') as f:
        pickle.dump({'X_frill': X_frill}, f)

    if errors:
        print(f"  [WARN] {len(errors)} file lỗi khi đọc audio.")
    print(f"  → Lưu cache: {CACHE_FRILL}")

print(f"  → Feature shape: {X_frill.shape}\n")

# ─── Chuẩn bị nhãn & Stratified K-Fold ───────────────────────────────────────
# Cập nhật mảng y cho XGBoost
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['label_respiratory'].values)
active_classes = df['label_respiratory'].unique()
num_active_classes = len(active_classes)
X = X_frill

# Tính class weight từ phân phối mẫu
class_counts = np.bincount(y, minlength=num_active_classes).astype(float)
total        = class_counts.sum()
# sample_weight tương ứng với mức độ hiếm của mỗi class
sample_weight = np.array([total / (num_active_classes * class_counts[yi]) for yi in y])

Fold  = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
folds = np.zeros(len(df), dtype=int)
for n, (_, val_idx) in enumerate(Fold.split(X, y)):
    folds[val_idx] = n

# ─── Huấn luyện & Đánh giá Cross-Validation ──────────────────────────────────
print("[INFO] Bắt đầu 5-Fold Cross-Validation...")
all_targets = []
all_preds   = []
all_probs   = []
fold_aucs   = []

for fold in range(FOLD_NUM):
    train_idx = folds != fold
    val_idx   = folds == fold

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    sw_train         = sample_weight[train_idx]

    model = xgb.XGBClassifier(
        objective       = 'multi:softprob',
        num_class       = num_active_classes,
        eval_metric     = 'mlogloss',
        max_depth       = 6,
        learning_rate   = 0.07,
        n_estimators    = 200,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        seed            = SEED,
        nthread         = -1,
        verbosity       = 0,
    )
    model.fit(X_train, y_train, sample_weight=sw_train)

    probs_val = model.predict_proba(X_val)  # shape (n_val, 5)
    preds_val = np.argmax(probs_val, axis=1)

    try:
        auc = roc_auc_score(y_val, probs_val, multi_class='ovo', average='macro')
    except Exception:
        auc = float('nan')

    fold_aucs.append(auc)
    all_targets.extend(y_val.tolist())
    all_preds.extend(preds_val.tolist())
    all_probs.extend(probs_val.tolist())

    print(f"  Fold {fold+1}/5 – AUC: {auc:.4f} | "
          f"Acc: {accuracy_score(y_val, preds_val):.4f}")

# ─── Tổng hợp kết quả ─────────────────────────────────────────────────────────
all_targets = np.array(all_targets)
all_preds   = np.array(all_preds)
all_probs   = np.array(all_probs)

print("\n" + "=" * 60)
print("  KẾT QUẢ TỔNG HỢP (5-Fold Cross-Validation)")
print("=" * 60)

mean_auc = np.nanmean(fold_aucs)
std_auc  = np.nanstd(fold_aucs)
macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
acc      = accuracy_score(all_targets, all_preds)

print(f"\n  Macro AUC   : {mean_auc:.4f} ± {std_auc:.4f}")
print(f"  Accuracy    : {acc:.4f}")
print(f"  Macro F1    : {macro_f1:.4f}")

print("\n  Confusion Matrix (hàng=Thực tế, cột=Dự đoán):")
cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_active_classes)))
active_names = [RESP_CLASSES[i] for i in le.classes_]
header = "          " + "  ".join(f"{name:>10}" for name in active_names)
print(header)
for i, row_vals in enumerate(cm):
    row_str = f"  {active_names[i]:>8}: " + "  ".join(f"{v:>10}" for v in row_vals)
    print(row_str)

print("\n  Classification Report:")
print(classification_report(
    all_targets, all_preds,
    target_names=active_names,
    zero_division=0
))

# ─── Lưu kết quả dự đoán ─────────────────────────────────────────────────────
result_df = df[['file_path', 'label_respiratory']].copy()
# Khôi phục nhãn gốc
all_preds_orig = le.inverse_transform(all_preds)
all_targets_orig = le.inverse_transform(all_targets)

result_df['pred_label']     = all_preds_orig
result_df['pred_class']     = [RESP_CLASSES[p] for p in all_preds_orig]
result_df['true_class']     = [RESP_CLASSES[t] for t in all_targets_orig]
result_df['correct']        = (all_targets_orig == all_preds_orig)

# Thêm xác suất từng class
for i, name in enumerate(active_names):
    result_df[f'prob_{name}'] = all_probs[:, i]

out_csv = os.path.join(OUTPUT_DIR, 'respiratory_predictions.csv')
result_df.to_csv(out_csv, index=False)
print(f"\n  → Kết quả dự đoán đã được lưu: {out_csv}")
print("\nDone!")
