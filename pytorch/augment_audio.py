#!/usr/bin/env python
# coding: utf-8
"""
Chiến lược 2 – Audio Data Augmentation cho các class hiếm.
Đọc sample WAV thuộc Asthma / Pneumonia từ data.csv,
sinh thêm file âm thanh qua 5 kỹ thuật augmentation,
cập nhật data.csv với các dòng mới, và trích xuất lại
FRILL embeddings (dùng cache) để chuẩn bị cho re-train.

Run:
  source .venv/bin/activate && python pytorch/augment_audio.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, copy, pickle, shutil
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

SOUNDDR_DIR  = os.path.join(os.path.dirname(__file__), '..', 'Sound-Dr')
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')   # ← thư mục gốc có cough/
DATA_DIR     = os.path.join(SOUNDDR_DIR, 'sounddr_data')
CSV_PATH     = os.path.join(DATA_DIR, 'data.csv')
AUG_WAV_DIR  = os.path.join(DATA_DIR, 'cough', 'augmented')
OUT_CSV      = os.path.join(DATA_DIR, 'data_augmented.csv')
CACHE_ORIG   = os.path.join(DATA_DIR, 'output', 'FRILL.pickle')
CACHE_AUG    = os.path.join(DATA_DIR, 'output', 'FRILL_augmented.pickle')

SR = 16000
os.makedirs(AUG_WAV_DIR, exist_ok=True)

# ─── Hàm gán nhãn ────────────────────────────────────────────────────────────
def get_label(row) -> int:
    cov  = str(row.get('cov19_status_choice', 'never')).lower()
    cond = str(row.get('medical_condition_choice', 'no')).lower()
    if cov in ('last14', 'over14'): return 4
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')): return 3
    if 'copd' in cond: return 2
    if 'asthma' in cond: return 1
    return 0

RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']
CLASS_NAMES  = {1: 'Asthma', 3: 'Pneumonia'}  # Chỉ augment class hiếm

TARGET_PER_CLASS = 80  # Mục tiêu mẫu của mỗi class hiếm sau augmentation

# ─── Đọc dữ liệu ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
if 'error' in df.columns:
    df = df[df['error'] == 0].reset_index(drop=True)

df['label_respiratory'] = df.apply(get_label, axis=1)
df['file_path'] = df['file_cough'].astype(str) + '.wav'

rare_df = df[df['label_respiratory'].isin(CLASS_NAMES.keys())].copy()
print(f"[INFO] Class hiếm tìm thấy:")
for cls_id, name in CLASS_NAMES.items():
    n = (rare_df['label_respiratory'] == cls_id).sum()
    print(f"  {name}: {n} mẫu gốc → mục tiêu: {TARGET_PER_CLASS} mẫu")

# ─── Các hàm augmentation bằng librosa (không cần GPU) ───────────────────────
def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def add_noise(y, snr_db=20):
    signal_power = np.mean(y ** 2)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(y))
    return np.clip(y + noise, -1.0, 1.0)

def pitch_shift_down(y, sr): return pitch_shift(y, sr, n_steps=-2)
def pitch_shift_up(y, sr):   return pitch_shift(y, sr, n_steps=+2)
def time_stretch_slow(y, sr): return time_stretch(y, rate=0.85)
def time_stretch_fast(y, sr): return time_stretch(y, rate=1.15)
def add_white_noise(y, sr):   return add_noise(y, snr_db=20)

AUG_FUNCS = {
    'ps_down': pitch_shift_down,
    'ps_up':   pitch_shift_up,
    'ts_slow': time_stretch_slow,
    'ts_fast': time_stretch_fast,
    'noise':   add_white_noise,
}

# ─── Sinh dữ liệu augmented ───────────────────────────────────────────────────
print("\n[INFO] Tiến hành augmentation...")
new_rows = []
aug_frill_feats = {}

for idx, row in rare_df.iterrows():
    # WAV gốc nằm ở PROJECT_ROOT/cough/<filename>.wav
    wav_filename = os.path.basename(row['file_cough']) + '.wav'
    src_path = os.path.join(PROJECT_ROOT, 'cough', wav_filename)
    if not os.path.exists(src_path):
        # Fallback to Sound-Dr folder when original files are already migrated.
        src_path = os.path.join(DATA_DIR, 'cough', wav_filename)
    if not os.path.exists(src_path):
        print(f"  [WARN] File không tồn tại: {src_path}")
        continue

    try:
        audio, _ = librosa.load(src_path, sr=SR, mono=True)
    except Exception as e:
        print(f"  [WARN] Không đọc được {src_path}: {e}")
        continue

    cls_id   = int(row['label_respiratory'])
    cls_name = CLASS_NAMES[cls_id]
    base     = os.path.splitext(os.path.basename(row['file_cough']))[0]

    for aug_name, aug_fn in AUG_FUNCS.items():
        try:
            aug_audio = aug_fn(audio, SR)
        except Exception as e:
            print(f"  [WARN] Augmentation {aug_name} lỗi: {e}")
            continue

        # Tạo tên file mới
        aug_filename = f"{base}_{aug_name}.wav"
        aug_path     = os.path.join(AUG_WAV_DIR, aug_filename)
        rel_path     = os.path.join('cough', 'augmented', aug_filename)

        # Lưu WAV
        sf.write(aug_path, aug_audio, SR)

        # Ghi dòng mới cho data_augmented.csv
        new_row = row.to_dict()
        new_row['file_cough']        = os.path.join('cough', 'augmented', os.path.splitext(aug_filename)[0])
        new_row['file_path']         = rel_path
        new_row['label_respiratory'] = cls_id
        new_row['augmented']         = True
        new_rows.append(new_row)

    print(f"  ✓ {cls_name} [{base}] – sinh {len(AUG_FUNCS)} mẫu mới")

# ─── Lưu CSV augmented ────────────────────────────────────────────────────────
aug_meta = pd.DataFrame(new_rows)
df['augmented'] = False
combined = pd.concat([df, aug_meta], ignore_index=True)
combined.to_csv(OUT_CSV, index=False)
print(f"\n[INFO] Phân phối sau augmentation:")
for cls_id, name in enumerate(RESP_CLASSES):
    n = (combined['label_respiratory'] == cls_id).sum()
    if n > 0:
        print(f"  {cls_id} – {name:12s}: {n:4d} mẫu")
print(f"  Tổng: {len(combined)} mẫu")
print(f"\n[INFO] Lưu CSV: {OUT_CSV}")

# ─── Trích xuất FRILL cho mẫu augmented ──────────────────────────────────────
print("\n[INFO] Tiến hành trích xuất FRILL cho mẫu augmented...")
print("  (Bỏ qua nếu không có TensorFlow – dùng librosa MFCC fallback...)")

try:
    sys.path.insert(0, os.path.abspath(SOUNDDR_DIR))
    from feature import make_nonsemantic_frill_nofrontend_feat
    USE_FRILL = True
    print("  → Dùng FRILL (TensorFlow Hub)")
except ImportError:
    USE_FRILL = False
    print("  → FRILL không khả dụng, dùng MFCC (128-dim) làm fallback")

def extract_mfcc(path, sr=SR, n_mfcc=128):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    except Exception:
        return np.zeros(n_mfcc * 2)

# Load existing FRILL cache
with open(CACHE_ORIG, 'rb') as f:
    cache = pickle.load(f)
X_orig = cache.get('X_trill_features', cache.get('X_frill'))

# Extract augmented features
aug_feats = []
for _, row in aug_meta.iterrows():
    aug_path = os.path.join(DATA_DIR, row['file_path'])
    if USE_FRILL:
        try:
            feat = make_nonsemantic_frill_nofrontend_feat(aug_path)
        except Exception:
            feat = np.zeros(X_orig.shape[1])
    else:
        feat = extract_mfcc(aug_path)
    aug_feats.append(feat)

aug_feats = np.array(aug_feats, dtype=np.float32)

if USE_FRILL:
    X_combined = np.vstack([X_orig, aug_feats])
    with open(CACHE_AUG, 'wb') as f:
        pickle.dump({'X_trill_features': X_combined,
                     'X_frill': X_combined}, f)
    print(f"\n[INFO] Lưu FRILL cache augmented: {CACHE_AUG}")
    print(f"  Feature shape: {X_combined.shape}")
else:
    # Fallback: lưu MFCC của các mẫu augmented riêng để dùng sau
    mfcc_path = os.path.join(DATA_DIR, 'output', 'MFCC_augmented.pickle')
    with open(mfcc_path, 'wb') as f:
        pickle.dump({'X_mfcc_aug': aug_feats,
                     'n_aug': len(aug_feats)}, f)
    print(f"\n[INFO] Lưu MFCC fallback: {mfcc_path}")
    print(f"  Shape: {aug_feats.shape}")
    print(f"\n  *** Để có FRILL embeddings đầy đủ, cần chạy lại khi TensorFlow hoạt động ***")

print("\nDone! Mẫu augmented được lưu tại:", AUG_WAV_DIR)
print("CSV mới:", OUT_CSV)
