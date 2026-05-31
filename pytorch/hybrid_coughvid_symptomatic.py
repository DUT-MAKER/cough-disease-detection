import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import importlib.util

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOUNDDR_DIR  = os.path.join(PROJECT_ROOT, 'Sound-Dr')
sys.path.insert(0, PROJECT_ROOT)

# Import FRILL from Sound-Dr/feature.py
_FEATURE_PY = os.path.join(SOUNDDR_DIR, 'feature.py')
_spec = importlib.util.spec_from_file_location("sounddr_feature", _FEATURE_PY)
_sounddr_feature = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_sounddr_feature)
    make_frill_feat = _sounddr_feature.make_nonsemantic_frill_nofrontend_feat
except Exception as e:
    print(f"[WARN] Cannot load FRILL feature module: {e}")
    make_frill_feat = None

# Import ResNet model
from pytorch.models import ResNet

# ---------------------------------------------------------
# HYPERPARAMETERS & SETTINGS
# ---------------------------------------------------------
CSV_PATH         = os.path.join(PROJECT_ROOT, 'public_dataset_v3', 'coughvid_covid_cough_vs_other_cough.csv')
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'workspace', 'hybrid_coughvid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_CNN        = os.path.join(OUTPUT_DIR, 'CNN_features_v3_symp.npy')
CACHE_FRILL      = os.path.join(OUTPUT_DIR, 'FRILL_features_v3_symp.npy')
RESNET_WEIGHT    = os.path.join(PROJECT_ROOT, 'save_resnet', 'best_model.pth')

SAMPLE_RATE      = 16000
MEL_BINS         = 64
HOP_LENGTH       = 256
N_FFT            = 512
MAX_FRAMES       = 626
NUM_CLASSES      = 2
FOLD_NUM         = 5
SEED             = 42

# ---------------------------------------------------------
# FEATURE EXTRACTORS
# ---------------------------------------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, class_num, weight_path):
        super().__init__()
        self.resnet = ResNet(class_num)
        if os.path.exists(weight_path):
            print(f"[INFO] Loading ResNet weights: {weight_path}")
            ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model', ckpt)
            # Remove 'module.' prefix if it exists
            state = {k.replace('module.', ''): v for k, v in state.items()}
            
            # Remove fc1 layer parameters to avoid size mismatch error (9 classes vs 2 classes)
            # We only use ResNet as a feature extractor (stopping before fc1), so this is safe.
            state.pop('fc1.weight', None)
            state.pop('fc1.bias', None)
            
            self.resnet.load_state_dict(state, strict=False)
            print("[INFO] ResNet weights loaded successfully (ignoring FC layer).")
        else:
            print(f"[WARN] Weight file {weight_path} not found. Using random initialization.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet.to(self.device)
        self.resnet.eval()

    def extract(self, logmel_tensor):
        """Extracts features from ResResNet before the final FC layer."""
        with torch.no_grad():
            x = logmel_tensor.to(self.device).float()
            # Follow the same forward logic as ResNet but stop before FC
            # Ensure input is 3D (batch, seq, mel)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            (batch_size, seq_len, mel_bins) = x.shape
            x = x.view(batch_size, 1, seq_len, mel_bins)
            x = x.transpose(1, 3)
            x = self.resnet.bn0(x)
            x = x.transpose(1, 3)
            x = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.resnet.resblock1(x)
            x = self.resnet.resblock2(x)
            x = F.max_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])
            return x.cpu().numpy()  # Return shape: (batch_size, 256)

def audio_to_logmel(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        if len(y) == 0: return np.zeros((MAX_FRAMES, MEL_BINS), dtype=np.float32)
    except Exception:
        return np.zeros((MAX_FRAMES, MEL_BINS), dtype=np.float32)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                        hop_length=HOP_LENGTH, n_mels=MEL_BINS)
    S_db = librosa.power_to_db(S, ref=np.max).T  # (T, 64)
    if S_db.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - S_db.shape[0], MEL_BINS), dtype=np.float32)
        S_db = np.vstack([S_db, pad])
    else:
        S_db = S_db[:MAX_FRAMES]
    return S_db.astype(np.float32)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print("  COUGHVID Hybrid Fusion: ResNet + FRILL + XGBoost")
    print("=" * 60)

    # 1. Load Clean Metadata
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Clean CSV not found at {CSV_PATH}. Please run prepare_coughvid.py first.")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Total clean samples: {len(df)}")
    print(df['status'].value_counts())
    print("\n[INFO] Label Mapping: 1=COVID-19, 0=Symptomatic")

    # 2. Extract / Load CNN Features
    if os.path.exists(CACHE_CNN):
        print(f"[INFO] Loading cached CNN features: {CACHE_CNN}")
        X_cnn = np.load(CACHE_CNN)
    else:
        print(f"[INFO] Extracting CNN Features (ResNet)...")
        extractor = ResNetFeatureExtractor(NUM_CLASSES, RESNET_WEIGHT)
        X_cnn = []
        batch_size = 32
        batch_tensors = []
        for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="CNN Extraction"):
            logmel = audio_to_logmel(row['file_path'])
            batch_tensors.append(logmel)
            
            if len(batch_tensors) == batch_size or idx == len(df) - 1:
                tensor = torch.tensor(np.array(batch_tensors))
                try:
                    feat_batch = extractor.extract(tensor)
                    X_cnn.extend(feat_batch)
                except Exception as e:
                    print(f"Error on batch ending at {row['uuid']}: {e}")
                    X_cnn.extend([np.zeros(256)] * len(batch_tensors))
                batch_tensors = []
                
        X_cnn = np.array(X_cnn, dtype=np.float32)
        np.save(CACHE_CNN, X_cnn)
        print(f"[INFO] Saved CNN features to {CACHE_CNN}")

    # 3. Extract / Load FRILL Features
    if os.path.exists(CACHE_FRILL):
        print(f"[INFO] Loading cached FRILL features: {CACHE_FRILL}")
        X_frill = np.load(CACHE_FRILL)
    elif make_frill_feat is not None:
        print(f"[INFO] Extracting FRILL Features (Acoustic)...")
        X_frill = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="FRILL Extraction"):
            try:
                feat = make_frill_feat(row['file_path'])
            except Exception as e:
                feat = np.zeros(2048 * 2) # Sound-Dr feature.py returns mean+std
            X_frill.append(feat)
        X_frill = np.array(X_frill, dtype=np.float32)
        np.save(CACHE_FRILL, X_frill)
        print(f"[INFO] Saved FRILL features to {CACHE_FRILL}")
    else:
        print("[ERROR] FRILL extraction is not available. Please check TensorFlow/Hub installation.")
        return

    # 4. Fusion
    print(f"\n[INFO] Fusing vectors: CNN ({X_cnn.shape[1]}) + FRILL ({X_frill.shape[1]})")
    X_fusion = np.hstack((X_frill, X_cnn))
    # 5. Prepare labels (0=Symptomatic, 1=COVID-19)
    # The new CSV already has 'label_binary' column mapped (1=COVID, 0=Symptomatic)
    y = df['label_binary'].values
    print(f"[INFO] Total Fusion vector shape: {X_fusion.shape}")

    # Handle nan/inf
    X_fusion = np.nan_to_num(X_fusion)

    # 5. Training with XGBoost (5-Fold CV)
    print(f"\n[INFO] Starting 5-Fold Cross Validation...")
    kf = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)
    
    all_targets, all_preds, all_probs, fold_aucs = [], [], [], []
    
    # Calculate scale_pos_weight for imbalance
    count_0 = (y == 0).sum()
    count_1 = (y == 1).sum()
    scale_weight = (count_0 / count_1) * 50
    print(f"  Class Distribution: 0={count_0}, 1={count_1} (Scale Weight: {scale_weight:.2f})")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_fusion, y)):
        clf = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',   # Tăng tốc độ bằng thuật toán Histogram
            device='cuda',        # Đẩy quá trình train lên thẳng GPU
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0
        )
        
        clf.fit(X_fusion[tr_idx], y[tr_idx])
        
        probs = clf.predict_proba(X_fusion[va_idx])[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        try:
            auc = roc_auc_score(y[va_idx], probs)
        except:
            auc = 0.5
            
        fold_aucs.append(auc)
        all_targets.extend(y[va_idx])
        all_preds.extend(preds)
        all_probs.extend(probs)
        
        print(f"  Fold {fold+1}: AUC={auc:.4f}, F1-Macro={f1_score(y[va_idx], preds, average='macro'):.4f}")

    # 6. Final Report
    print("\n" + "="*60)
    print("  FINAL PERFORMANCE REPORT (COUGHVID v3)")
    print("="*60)
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Symptomatic', 'COVID-19']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    # Save final model
    model_path = os.path.join(OUTPUT_DIR, 'coughvid_hybrid_xgb_symp.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'target_names': ['Symptomatic', 'COVID-19']}, f)
    print(f"\n[INFO] Final model saved to {model_path}")

if __name__ == '__main__':
    main()
