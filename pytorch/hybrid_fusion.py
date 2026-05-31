import warnings
warnings.filterwarnings("ignore")
import os, sys

# Thiết lập đường dẫn - PHẢI trước mọi import khác
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOUNDDR_DIR  = os.path.join(PROJECT_ROOT, 'Sound-Dr')
# PROJECT_ROOT trước, SOUNDDR_DIR sau để tránh Sound-Dr/utils.py ghi đè utils/
sys.path.insert(0, PROJECT_ROOT)
# KHÔNG thêm SOUNDDR_DIR vào sys.path – import feature.py bằng importlib riêng

import pickle, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import xgboost as xgb
import librosa
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

# ── Import Feature.py bằng importlib (tránh path conflict) ───────────────────
_FEATURE_PY = os.path.join(SOUNDDR_DIR, 'feature.py')
_spec = importlib.util.spec_from_file_location("sounddr_feature", _FEATURE_PY)
_sounddr_feature = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_sounddr_feature)
    make_frill_feat = _sounddr_feature.make_nonsemantic_frill_nofrontend_feat
except Exception as e:
    print(f"[WARN] Không load được FRILL feature: {e}")
    make_frill_feat = None

# Import model từ pytorch/models.py (dùng sys.path PROJECT_ROOT sẵn có)
from pytorch.models import ResNet

# ─── Cấu hình ─────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(SOUNDDR_DIR, 'sounddr_data')
CSV_PATH      = os.path.join(DATA_DIR, 'data.csv')
OUTPUT_DIR    = os.path.join(DATA_DIR, 'output_hybrid')
CACHE_FRILL   = os.path.join(DATA_DIR, 'output', 'FRILL.pickle')
CACHE_CNN     = os.path.join(DATA_DIR, 'output', 'CNN_hybrid.pickle')
RESNET_WEIGHT = os.path.join(PROJECT_ROOT, 'workspace', 'resnet_hybrid', 'best_model.pth')

SAMPLE_RATE  = 16000
MEL_BINS     = 64
HOP_LENGTH   = 256
N_FFT        = 512
MAX_FRAMES   = 626   # ≈ 10s @ 16 kHz với hop=256
FOLD_NUM     = 5
SEED         = 2022
RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']
NUM_CLASSES  = 2   # Binary: Healthy vs COVID
KEEP_LABELS  = {0: 0, 4: 1}   # Healthy→0, COVID→1
CLASS_NAMES  = ['Healthy', 'COVID']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'output'), exist_ok=True)

# ─── CNN Feature Extractor ─────────────────────────────────────────────────────
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, class_num, weight_path):
        super().__init__()
        self.resnet = ResNet(class_num)
        if os.path.exists(weight_path):
            print(f"[INFO] Tải trọng số CNN: {weight_path}")
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt)
            state = {k.replace('module.', ''): v for k, v in state.items()}
            self.resnet.load_state_dict(state, strict=False)
        else:
            print(f"[WARN] Không tìm thấy {weight_path}. Dùng trọng số ngẫu nhiên.")
        self.resnet.eval()

    def extract(self, logmel_tensor):
        """logmel_tensor: (1, seq_len, mel_bins) float32"""
        with torch.no_grad():
            (_, seq_len, mel_bins) = logmel_tensor.shape
            x = logmel_tensor.view(-1, 1, seq_len, mel_bins)
            x = x.transpose(1, 3)
            x = self.resnet.bn0(x)
            x = x.transpose(1, 3)
            x = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.resnet.resblock1(x)
            x = self.resnet.resblock2(x)
            x = F.max_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])
            return x.numpy().flatten()

# ─── Hàm tiện ích ─────────────────────────────────────────────────────────────
def get_label(row) -> int:
    cov  = str(row.get('cov19_status_choice', 'never')).strip().lower()
    cond = str(row.get('medical_condition_choice', "['No']")).lower()
    if cov in ('last14', 'over14'): return 4
    if any(k in cond for k in ('pneumonia', 'lung', 'pulmonary', 'cystic')): return 3
    if 'copd'   in cond: return 2
    if 'asthma' in cond: return 1
    return 0

def audio_to_logmel(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return np.zeros((MAX_FRAMES, MEL_BINS), dtype=np.float32)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                        hop_length=HOP_LENGTH, n_mels=MEL_BINS)
    S_db = librosa.power_to_db(S, ref=np.max).T  # (T, 64)
    # Padding / truncation
    if S_db.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - S_db.shape[0], MEL_BINS), dtype=np.float32)
        S_db = np.vstack([S_db, pad])
    else:
        S_db = S_db[:MAX_FRAMES]
    return S_db.astype(np.float32)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Mô hình Lai Kép – Hybrid Vision & Acoustic Fusion")
    print("=" * 60)

    # 1. Đọc metadata
    df = pd.read_csv(CSV_PATH)
    df['file_path']          = df['file_cough'] + '.wav'
    df['label_respiratory']  = df.apply(get_label, axis=1)
    if 'error' in df.columns:
        df = df[df['error'] == 0].reset_index(drop=True)
    print(f"[INFO] Số mẫu hợp lệ: {len(df)}")
    print("[INFO] Phân phối nhãn (5 lớp gốc):")
    for idx, cnt in df['label_respiratory'].value_counts().sort_index().items():
        print(f"  {idx} – {RESP_CLASSES[idx]:12s}: {cnt}")


    # 2. Lấy đặc trưng FRILL (từ cache)
    print("\n[INFO] Bước 1/2 – Đặc trưng FRILL (2048-d)...")
    if os.path.exists(CACHE_FRILL):
        print(f"  → Dùng cache: {CACHE_FRILL}")
        with open(CACHE_FRILL, 'rb') as f:
            cache_data = pickle.load(f)
        # Hỗ trợ nhiều key name
        X_frill = cache_data.get('X_trill_features', None)
        if X_frill is None:
            X_frill = cache_data.get('X_frill', None)
        if X_frill is None:
            # Fallback: lấy giá trị đầu tiên trong dict
            X_frill = next(iter(cache_data.values()))
    elif make_frill_feat is not None:
        X_frill = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="FRILL"):
            audio_path = os.path.join(DATA_DIR, row['file_path'])
            try:
                feat = make_frill_feat(audio_path)
            except Exception:
                feat = np.zeros(2048)
            X_frill.append(feat)
        X_frill = np.array(X_frill, dtype=np.float32)
        with open(CACHE_FRILL, 'wb') as f:
            pickle.dump({'X_frill': X_frill}, f)
    else:
        print("  [WARN] FRILL không khả dụng. Dùng MFCC fallback (39-d).")
        X_frill = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="MFCC"):
            ap = os.path.join(DATA_DIR, row['file_path'])
            try:
                y, sr = librosa.load(ap, sr=SAMPLE_RATE, mono=True)
                mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                feat  = np.concatenate([mfcc.mean(1), mfcc.std(1), mfcc.max(1)])
            except Exception:
                feat = np.zeros(39)
            X_frill.append(feat)
        X_frill = np.array(X_frill, dtype=np.float32)

    X_frill = np.nan_to_num(np.array(X_frill, dtype=np.float32))
    print(f"  → Shape FRILL: {X_frill.shape}")

    # 3. Lấy đặc trưng CNN Log-Mel (luồng mới)
    print("\n[INFO] Bước 2/2 – Đặc trưng CNN Log-Mel (256-d)...")
    if os.path.exists(CACHE_CNN):
        print(f"  → Dùng cache: {CACHE_CNN}")
        with open(CACHE_CNN, 'rb') as f:
            X_cnn = pickle.load(f)['X_cnn']
    else:
        extractor = ResNetFeatureExtractor(NUM_CLASSES, RESNET_WEIGHT)
        X_cnn = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CNN"):
            ap      = os.path.join(DATA_DIR, row['file_path'])
            logmel  = audio_to_logmel(ap)                        # (T, 64)
            tensor  = torch.tensor(logmel).unsqueeze(0)          # (1, T, 64)
            try:
                feat = extractor.extract(tensor)
            except Exception:
                feat = np.zeros(256)
            X_cnn.append(feat)
        X_cnn = np.array(X_cnn, dtype=np.float32)
        with open(CACHE_CNN, 'wb') as f:
            pickle.dump({'X_cnn': X_cnn}, f)
        print(f"  → Cache CNN đã lưu: {CACHE_CNN}")

    print(f"  → Shape CNN: {X_cnn.shape}")

    # 4. Fusion: ghép vector
    X_fusion = np.hstack((X_frill, X_cnn))
    
    # Lọc nhãn: Chỉ giữ Healthy (0) và COVID (4)
    valid_mask = df['label_respiratory'].isin(KEEP_LABELS.keys())
    X_fusion = X_fusion[valid_mask]
    df = df[valid_mask].reset_index(drop=True)
    df['label_binary'] = df['label_respiratory'].map(KEEP_LABELS)
    
    print(f"\n[INFO] Sau khi lọc Healthy + COVID: {len(df)} mẫu")
    for lbl, name in enumerate(CLASS_NAMES):
        print(f"  {lbl} – {name:10s}: {(df['label_binary']==lbl).sum()}")
    
    print(f"\n[INFO] Hybrid Super-Vector: {X_frill.shape[1]}-d FRILL + {X_cnn.shape[1]}-d CNN"
          f" = {X_fusion.shape[1]}-d")

    # 5. Chuẩn bị nhãn & sample_weight (dùng label_binary)
    y            = df['label_binary'].values
    n_cls        = NUM_CLASSES   # 2
    cls_counts   = np.bincount(y, minlength=n_cls).astype(float)
    sw           = np.array([cls_counts.sum() / (n_cls * cls_counts[yi]) for yi in y])
    active_names = CLASS_NAMES

    # 6. 5-Fold Cross-Validation trên XGBoost
    print(f"\n[INFO] 5-Fold CV – XGBoost trên {X_fusion.shape[1]}-d Hybrid vector...")
    kf           = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
    all_targets, all_preds, all_probs, fold_aucs = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_fusion, y)):
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6, learning_rate=0.07, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            seed=SEED, nthread=-1, verbosity=0
        )
        clf.fit(X_fusion[tr_idx], y[tr_idx], sample_weight=sw[tr_idx])
        probs_1d = clf.predict_proba(X_fusion[va_idx])[:, 1]  # P(COVID)
        preds    = (probs_1d >= 0.5).astype(int)
        probs    = clf.predict_proba(X_fusion[va_idx])
        try:
            auc = roc_auc_score(y[va_idx], probs_1d)
        except Exception:
            auc = float('nan')
        fold_aucs.append(auc)
        all_targets.extend(y[va_idx].tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        print(f"  Fold {fold+1}/5 – AUC: {auc:.4f} | "
              f"Acc: {accuracy_score(y[va_idx], preds):.4f} | "
              f"F1 : {f1_score(y[va_idx], preds, average='macro', zero_division=0):.4f}")

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)

    # 7. In kết quả tổng hợp
    print("\n" + "=" * 60)
    print("  KẾT QUẢ MÔ HÌNH LAI KÉP (Nghiên cứu Khoa Học)")
    print("=" * 60)
    print(f"  Macro AUC : {np.nanmean(fold_aucs):.4f} ± {np.nanstd(fold_aucs):.4f}")
    print(f"  Accuracy  : {accuracy_score(all_targets, all_preds):.4f}")
    print(f"  Macro F1  : {f1_score(all_targets, all_preds, average='macro', zero_division=0):.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=active_names, zero_division=0))

    # 8. Lưu model production
    print("[INFO] Lưu model production (train toàn bộ dữ liệu)...")
    final = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=6, learning_rate=0.07, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8,
        seed=SEED, nthread=-1, verbosity=0
    )
    final.fit(X_fusion, y, sample_weight=sw)
    with open(os.path.join(OUTPUT_DIR, 'xgb_hybrid_final.pkl'), 'wb') as f:
        pickle.dump({'model': final, 'class_names': CLASS_NAMES,
                     'cnn_dim': X_cnn.shape[1], 'frill_dim': X_frill.shape[1]}, f)
    print(f"  → Model lưu tại: {OUTPUT_DIR}/xgb_hybrid_final.pkl")
    print("\nDone!")

if __name__ == '__main__':
    main()
