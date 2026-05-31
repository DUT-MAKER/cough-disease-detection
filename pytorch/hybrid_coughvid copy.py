import argparse
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

# .webm/.ogg: soundfile thường không đọc được → librosa fallback audioread (spam UserWarning + FutureWarning).
warnings.filterwarnings('ignore', message=r'PySoundFile failed\..*', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'librosa\.core\.audio')

from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                             confusion_matrix)
import importlib.util


def tune_threshold_f1_macro(y_true, scores, n_steps=99):
    """Tìm ngưỡng trên xác suất lớp dương để tối đa hóa F1-macro (trên tập validation của fold)."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOUNDDR_DIR  = os.path.join(PROJECT_ROOT, 'Sound-Dr')
sys.path.insert(0, PROJECT_ROOT)

# TensorFlow (FRILL): dùng GPU nếu TF build hỗ trợ CUDA; kiểm tra chặt ở PyTorch phía trên.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

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


def require_torch_cuda_device() -> torch.device:
    """Ép buộc PyTorch có CUDA; toàn bộ ResNet/fine-tune chạy trên GPU."""
    if not torch.cuda.is_available():
        print(
            '[ERROR] Không có GPU CUDA cho PyTorch. Script này bắt buộc chạy trên GPU '
            '(ResNet + XGBoost tree_method hist trên CUDA).\n'
            '  Cài driver NVIDIA + PyTorch build có CUDA, rồi chạy lại.'
        )
        raise SystemExit(1)
    d = torch.device('cuda:0')
    p = torch.cuda.get_device_properties(0)
    print(f'[INFO] GPU bắt buộc: {p.name} ({p.total_memory / (1024 ** 3):.1f} GiB)')
    return d


# ---------------------------------------------------------
# HYPERPARAMETERS & SETTINGS
# ---------------------------------------------------------
DEFAULT_CSV_PATH = os.path.join(
    PROJECT_ROOT,
    'public_dataset_v3',
    'coughvid_covid_cough_vs_other_cough.csv',
)
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, 'workspace', 'hybrid_coughvid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESNET_WEIGHT    = os.path.join(PROJECT_ROOT, 'save_resnet', 'best_model.pth')

SAMPLE_RATE      = 16000
MEL_BINS         = 64
HOP_LENGTH       = 256
N_FFT            = 512
MAX_FRAMES       = 626
NUM_CLASSES      = 2
FOLD_NUM         = 5
SEED             = 42

FT_BATCH_DEFAULT  = 128 * 4  # mặc định khi không truyền CLI; fine-tune + embedding dùng --batch-size
FT_LR              = 3e-4
FT_VAL_SIZE        = 0.12


def artifact_paths_for_csv(csv_path: str) -> dict:
    """Cache / checkpoint tách theo tên CSV để không lẫn số mẫu (vd. 12k vs 4k)."""
    stem = os.path.splitext(os.path.basename(os.path.abspath(csv_path)))[0]
    return {
        'stem': stem,
        'cache_logmel': os.path.join(OUTPUT_DIR, f'logmel_{stem}.npy'),
        'cache_cnn': os.path.join(OUTPUT_DIR, f'CNN_{stem}.npy'),
        'cache_frill': os.path.join(OUTPUT_DIR, f'FRILL_{stem}.npy'),
        'cache_cnn_ft': os.path.join(OUTPUT_DIR, f'CNN_{stem}_finetuned.npy'),
        'finetuned_ckpt': os.path.join(OUTPUT_DIR, f'resnet_{stem}_finetuned.pt'),
        'model_pkl': os.path.join(OUTPUT_DIR, f'hybrid_xgb_{stem}.pkl'),
    }


def target_names_for_csv_stem(stem: str) -> list:
    """Nhãn lớp 0 / 1 khớp label_binary trong CSV."""
    if stem == 'coughvid_clean':
        return ['Healthy', 'COVID-19']
    return ['Symptomatic_ho', 'COVID-19_ho']


# ---------------------------------------------------------
# FEATURE EXTRACTORS
# ---------------------------------------------------------
def _load_resnet_state_dict(weight_path):
    ckpt = torch.load(weight_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    return {k.replace('module.', ''): v for k, v in state.items()}


@torch.no_grad()
def batch_resnet_embeddings(resnet, X, device, batch_size=128):
    """X: (N, T, mel) float tensor trên CPU → (N, 256) numpy."""
    resnet.eval()
    chunks = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i : i + batch_size].to(device).float()
        (_, seq_len, mel_bins) = xb.shape
        x = xb.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = resnet.bn0(x)
        x = x.transpose(1, 3)
        x = F.relu(resnet.bn1(resnet.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = resnet.resblock1(x)
        x = resnet.resblock2(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        chunks.append(x.cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, class_num, weight_path, finetuned=False, device=None):
        super().__init__()
        if device is None:
            device = require_torch_cuda_device()
        self.resnet = ResNet(class_num)
        if os.path.exists(weight_path):
            print(f"[INFO] Loading ResNet weights: {weight_path} (finetuned={finetuned})")
            state = _load_resnet_state_dict(weight_path)
            if finetuned:
                self.resnet.load_state_dict(state, strict=True)
                print("[INFO] ResNet fine-tuned checkpoint loaded (full state).")
            else:
                state.pop('fc1.weight', None)
                state.pop('fc1.bias', None)
                self.resnet.load_state_dict(state, strict=False)
                print("[INFO] ResNet pretrained loaded (FC layer re-init / ignored).")
        else:
            print(f"[WARN] Weight file {weight_path} not found. Using random initialization.")
        self.device = device
        self.resnet.to(self.device)
        self.resnet.eval()

    def extract(self, logmel_tensor):
        """Extracts features from ResResNet before the final FC layer."""
        with torch.no_grad():
            x = logmel_tensor.to(self.device).float()
            # Follow the same forward logic as ResNet but stop before FC
            (_, seq_len, mel_bins) = x.shape
            x = x.view(-1, 1, seq_len, mel_bins)
            x = x.transpose(1, 3)
            x = self.resnet.bn0(x)
            x = x.transpose(1, 3)
            x = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.resnet.resblock1(x)
            x = self.resnet.resblock2(x)
            x = F.max_pool2d(x, kernel_size=x.shape[2:])
            x = x.view(x.shape[0:2])
            return x.cpu().numpy().flatten()

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


def load_all_logmels(df):
    """(N, MAX_FRAMES, MEL_BINS) float32 — dùng cho fine-tune và embedding hàng loạt."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Load log-mel (RAM)"):
        rows.append(audio_to_logmel(row['file_path']))
    return np.stack(rows, axis=0).astype(np.float32)


def load_or_build_logmel_cache(
    df,
    cache_path: str,
    csv_path: str,
    force_rebuild: bool,
) -> np.ndarray:
    """Tránh đọc lại toàn bộ audio mỗi lần fine-tune: dùng logmel_*.npy nếu còn hợp lệ."""
    n = len(df)
    csv_path = os.path.abspath(csv_path)
    cache_ok = (
        not force_rebuild
        and os.path.isfile(cache_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(csv_path)
    )
    if cache_ok:
        X = np.load(cache_path)
        if X.shape == (n, MAX_FRAMES, MEL_BINS):
            print(
                f"[INFO] Dùng cache log-mel ({n} mẫu, trùng CSV): {cache_path}"
            )
            return X.astype(np.float32, copy=False)
        print("[WARN] Cache log-mel lệch shape/số dòng — tính lại từ audio.")
    else:
        if force_rebuild:
            print("[INFO] --force-reload-logmel: bỏ qua cache log-mel.")
        elif os.path.isfile(cache_path):
            print("[INFO] CSV mới hơn cache log-mel — tính lại từ audio.")

    X = load_all_logmels(df)
    np.save(cache_path, X)
    print(f"[INFO] Đã lưu log-mel cache → {cache_path}")
    return X


def finetune_resnet_binary(
    X_np,
    y_np,
    pretrained_path,
    save_ckpt_path,
    epochs,
    device,
    batch_size,
):
    """Fine-tune nhẹ: đóng băng tới hết resblock1, mở resblock2 + fc1. Lưu checkpoint 2 lớp."""
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np.astype(np.int64))
    idx = np.arange(len(y))
    tr_idx, va_idx = train_test_split(
        idx, test_size=FT_VAL_SIZE, stratify=y_np, random_state=SEED
    )
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    model = ResNet(NUM_CLASSES).to(device)
    state = _load_resnet_state_dict(pretrained_path)
    state.pop('fc1.weight', None)
    state.pop('fc1.bias', None)
    _incomp = model.load_state_dict(state, strict=False)
    if _incomp is not None:
        uk = getattr(_incomp, 'unexpected_keys', None) or []
        if uk:
            print(f"[WARN] Unexpected keys khi load pretrained: {uk}")

    for mod in (model.bn0, model.conv1, model.bn1, model.resblock1):
        for p in mod.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=FT_LR, weight_decay=1e-4)

    n0 = int((y_tr == 0).sum().item())
    n1 = int((y_tr == 1).sum().item())
    w = torch.tensor(
        [max(len(y_tr) / max(n0, 1), 1e-6), max(len(y_tr) / max(n1, 1), 1e-6)],
        device=device,
        dtype=torch.float32,
    )
    w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=w)

    bs = max(1, int(batch_size))
    dl_tr = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=bs,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    dl_va = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=min(bs * 2, 512),
        shuffle=False,
        num_workers=0,
    )

    best_auc = 0.0
    best_state = None
    ckpt_dir = os.path.dirname(save_ckpt_path)
    ckpt_stem = os.path.splitext(os.path.basename(save_ckpt_path))[0]

    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        probs, y_true = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                pr = F.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
                probs.append(pr)
                y_true.append(yb.numpy())
        probs = np.concatenate(probs)
        y_true = np.concatenate(y_true)
        try:
            va_auc = roc_auc_score(y_true, probs)
        except Exception:
            va_auc = 0.5

        print(
            f"  [FT] epoch {ep + 1}/{epochs}  loss={np.mean(losses):.4f}  val_AUC={va_auc:.4f}"
        )
        ep_path = os.path.join(ckpt_dir, f'{ckpt_stem}_ep{ep + 1:02d}.pt')
        torch.save(
            {
                'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'val_auc': float(va_auc),
                'epoch': ep + 1,
            },
            ep_path,
        )
        if va_auc >= best_auc:
            best_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    # File chính: weights tốt nhất theo val AUC (dùng cho bước embedding + pipeline sau)
    torch.save({'model': model.state_dict(), 'val_auc': best_auc}, save_ckpt_path)
    print(
        f"[INFO] Fine-tune xong. Best val AUC≈{best_auc:.4f} → {save_ckpt_path} "
        f"(+ mỗi epoch: {ckpt_stem}_ep01.pt …)"
    )
    return model


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    art = artifact_paths_for_csv(csv_path)
    target_names = target_names_for_csv_stem(art['stem'])

    print("=" * 60)
    print("  COUGHVID Hybrid Fusion: ResNet + FRILL + XGBoost")
    print(f"  CSV: {art['stem']}  |  Lớp 0/1: {target_names[0]} / {target_names[1]}")
    print("=" * 60)

    device = require_torch_cuda_device()

    try:
        import tensorflow as tf

        _tf_gpus = tf.config.list_physical_devices('GPU')
        if _tf_gpus:
            print(f'[INFO] TensorFlow (FRILL): {len(_tf_gpus)} GPU.')
        else:
            print(
                '[WARN] TensorFlow không thấy GPU — FRILL có thể chạy CPU; '
                'cài tensorflow[and-cuda] hoặc build TF-GPU nếu cần tăng tốc FRILL.'
            )
    except Exception as _e:
        print(f'[WARN] Không kiểm tra được TensorFlow: {_e}')

    # 1. Load metadata
    if not os.path.exists(csv_path):
        print(
            f"[ERROR] Không thấy CSV: {csv_path}\n"
            "  Chạy prepare_coughvid_cough_covid_vs_other.py hoặc prepare_coughvid.py, "
            "hoặc truyền --csv đúng đường dẫn."
        )
        return

    df = pd.read_csv(csv_path)
    if 'label_binary' not in df.columns:
        print("[ERROR] CSV phải có cột label_binary (0/1).")
        return
    if 'file_path' not in df.columns:
        print("[ERROR] CSV phải có cột file_path.")
        return
    print(f"[INFO] Số mẫu: {len(df)}")
    if 'status' in df.columns:
        print(df['status'].value_counts())

    use_finetune = args.finetune_epochs > 0
    cnn_cache_path = art['cache_cnn_ft'] if use_finetune else art['cache_cnn']
    finetuned_ckpt_path = art['finetuned_ckpt']
    resnet_weights_for_extract = finetuned_ckpt_path if use_finetune else RESNET_WEIGHT

    # 2. CNN: fine-tune (tùy chọn) + embedding 256 chiều
    if os.path.exists(cnn_cache_path) and not args.force_finetune:
        print(f"[INFO] Loading cached CNN features: {cnn_cache_path}")
        X_cnn = np.load(cnn_cache_path)
    elif use_finetune:
        need_train = args.force_finetune or (not os.path.exists(finetuned_ckpt_path))
        print(
            f"\n[INFO] Chuẩn bị fine-tune ResNet ({args.finetune_epochs} epoch, batch_size={args.batch_size})..."
        )
        X_mel = load_or_build_logmel_cache(
            df,
            art['cache_logmel'],
            csv_path,
            args.force_reload_logmel,
        )
        y_arr = df['label_binary'].values.astype(np.int64)
        if need_train:
            if not os.path.exists(RESNET_WEIGHT):
                print(f"[ERROR] Thiếu pretrained ResNet: {RESNET_WEIGHT}")
                return
            finetune_resnet_binary(
                X_mel,
                y_arr,
                RESNET_WEIGHT,
                finetuned_ckpt_path,
                args.finetune_epochs,
                device,
                args.batch_size,
            )
        ckpt = torch.load(finetuned_ckpt_path, map_location=device, weights_only=False)
        model_ft = ResNet(NUM_CLASSES).to(device)
        model_ft.load_state_dict(ckpt['model'], strict=True)
        embed_bs = min(max(args.batch_size, 1) * 4, 512)
        X_cnn = batch_resnet_embeddings(
            model_ft,
            torch.from_numpy(X_mel),
            device,
            batch_size=embed_bs,
        )
        del X_mel, model_ft
        torch.cuda.empty_cache()
        np.save(cnn_cache_path, X_cnn)
        print(f"[INFO] Saved CNN features (fine-tuned backbone) → {cnn_cache_path}")
    else:
        if os.path.exists(art['cache_cnn']):
            print(f"[INFO] Loading cached CNN features: {art['cache_cnn']}")
            X_cnn = np.load(art['cache_cnn'])
        else:
            print("[INFO] Extracting CNN Features (ResNet pretrained, không fine-tune)...")
            extractor = ResNetFeatureExtractor(
                NUM_CLASSES, RESNET_WEIGHT, finetuned=False, device=device
            )
            X_cnn = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="CNN Extraction"):
                logmel = audio_to_logmel(row['file_path'])
                tensor = torch.tensor(logmel).unsqueeze(0)
                try:
                    feat = extractor.extract(tensor)
                except Exception as e:
                    print(f"Error on {row['uuid']}: {e}")
                    feat = np.zeros(256)
                X_cnn.append(feat)
            X_cnn = np.array(X_cnn, dtype=np.float32)
            np.save(art['cache_cnn'], X_cnn)
            print(f"[INFO] Saved CNN features to {art['cache_cnn']}")

    # 3. Extract / Load FRILL Features
    if os.path.exists(art['cache_frill']):
        print(f"[INFO] Loading cached FRILL features: {art['cache_frill']}")
        X_frill = np.load(art['cache_frill'])
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
        np.save(art['cache_frill'], X_frill)
        print(f"[INFO] Saved FRILL features to {art['cache_frill']}")
    else:
        print("[ERROR] FRILL extraction is not available. Please check TensorFlow/Hub installation.")
        return

    # 4. Fusion
    print(f"\n[INFO] Fusing vectors: CNN ({X_cnn.shape[1]}) + FRILL ({X_frill.shape[1]})")
    X_fusion = np.hstack((X_frill, X_cnn))
    y = df['label_binary'].values
    print(f"[INFO] Total Fusion vector shape: {X_fusion.shape}")

    # Handle nan/inf
    X_fusion = np.nan_to_num(X_fusion)

    # 5. Training with XGBoost (5-Fold CV) + tune threshold per fold
    print(f"\n[INFO] Starting 5-Fold Cross Validation (threshold tuned on each val fold)...")
    kf = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)

    all_targets, all_preds, all_probs, fold_aucs = [], [], [], []
    fold_thresholds = []

    count_0 = int((y == 0).sum())
    count_1 = int((y == 1).sum())
    scale_weight = count_0 / max(count_1, 1)
    print(f"  Class Distribution: 0={count_0}, 1={count_1} (scale_pos_weight={scale_weight:.4f})")

    xgb_device = 'cuda'

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_fusion, y)):
        clf_fold = xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            device=xgb_device,
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )

        clf_fold.fit(X_fusion[tr_idx], y[tr_idx])

        probs = clf_fold.predict_proba(X_fusion[va_idx])[:, 1]
        y_va = y[va_idx]
        best_t, best_f1_on_grid = tune_threshold_f1_macro(y_va, probs)
        fold_thresholds.append(best_t)
        preds = (probs >= best_t).astype(int)

        try:
            auc = roc_auc_score(y_va, probs)
        except Exception:
            auc = 0.5

        fold_aucs.append(auc)
        all_targets.extend(y_va.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

        f1m = f1_score(y_va, preds, average='macro', zero_division=0)
        print(
            f"  Fold {fold + 1}: AUC={auc:.4f}, F1-Macro={f1m:.4f}, "
            f"threshold={best_t:.4f} (grid best F1-macro={best_f1_on_grid:.4f})"
        )

    threshold_agg = float(np.median(fold_thresholds))

    # 6. Final Report (OOF với ngưỡng theo từng fold)
    print("\n" + "=" * 60)
    print(f"  FINAL PERFORMANCE REPORT ({art['stem']})")
    print("=" * 60)
    print(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"Per-fold thresholds: {[round(t, 4) for t in fold_thresholds]}")
    print(f"Median threshold (production gợi ý): {threshold_agg:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    # 7. Huấn luyện mô hình cuối trên toàn bộ dữ liệu + lưu kèm ngưỡng
    print("\n[INFO] Training final XGBoost on full data for deployment...")
    clf_final = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        device=xgb_device,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    clf_final.fit(X_fusion, y)

    model_path = art['model_pkl']
    pos_name = target_names[1]
    bundle = {
        'model': clf_final,
        'decision_threshold': threshold_agg,
        'fold_thresholds': fold_thresholds,
        'scale_pos_weight': scale_weight,
        'target_names': target_names,
        'positive_class_index': 1,
        'csv_path': csv_path,
        'csv_stem': art['stem'],
        'resnet_finetuned': use_finetune,
        'resnet_ckpt': resnet_weights_for_extract if use_finetune else None,
        'cnn_cache': cnn_cache_path,
        'logmel_cache': art['cache_logmel'],
        'batch_size': args.batch_size,
        'note': (
            f'Proba cột 1 = P({pos_name}); dự đoán 1 nếu proba >= decision_threshold.'
        ),
    }
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[INFO] Final model saved to {model_path} (full-data fit + median CV threshold)")

def _positive_int(name):
    def _coerce(x):
        v = int(x)
        if v < 1:
            raise argparse.ArgumentTypeError(f'{name} phải >= 1')
        return v

    return _coerce


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Hybrid CoughVID: ResNet (+ fine-tune) + FRILL + XGBoost. '
            'Bắt buộc CUDA (PyTorch + XGBoost trên GPU). Đọc mel (librosa) vẫn trên CPU.'
        )
    )
    p.add_argument(
        '--csv',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='CSV metadata (cột file_path, label_binary). Mặc định: ho COVID vs ho symptomatic.',
    )
    p.add_argument(
        '--finetune-epochs',
        type=int,
        default=5,
        help='Số epoch fine-tune ResNet (resblock2+fc1). 0 = tắt, dùng pretrained như cũ.',
    )
    p.add_argument(
        '--force-finetune',
        action='store_true',
        help='Train lại ResNet và làm mới embedding cache dù đã có file.',
    )
    p.add_argument(
        '--force-reload-logmel',
        action='store_true',
        help='Bỏ qua file logmel_*.npy, đọc lại toàn bộ audio (chậm).',
    )
    p.add_argument(
        '--batch-size',
        type=_positive_int('--batch-size'),
        default=FT_BATCH_DEFAULT,
        metavar='N',
        help='Batch fine-tune ResNet (train). Validation = min(2N, 512); embedding = min(4N, 512). Mặc định %(default)s.',
    )
    return p.parse_args()


if __name__ == '__main__':
    main()
