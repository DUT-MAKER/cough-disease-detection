"""
predict_audio.py — Classify a single audio file into one of 9 sound categories
using the CovNet / FluSense backbone models.

Usage:
    python predict_audio.py --audio_file <path_to_audio> \
                            --model_path save_model/best_model.pth \
                            --backbone baseline

Supported audio formats: .wav, .flac (mp3 requires additional codec)
"""

import os
import sys
import argparse
import math

import math
import numpy as np
import torch
import torch.nn as nn

# ── Make sure sibling packages are importable ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import soundfile
from scipy import signal
from scipy.signal import resample_poly
from math import gcd

# Project imports
from utils.config import (
    sample_rate, audio_samples_flusense,
    window_size, overlap, mel_bins,
    classes_num_flusense,
)

# ── 9 FluSense sound labels (matches training label encoding) ──────────────────
LABELS = [
    'cough',            # 0
    'speech',           # 1
    'etc',              # 2
    'silence',          # 3
    'sneeze',           # 4
    'gasp',             # 5
    'breathe',          # 6
    'sniffle',          # 7
    'throat-clearing',  # 8
]

LABEL_VI = {
    'cough':            'Tiếng ho',
    'speech':           'Tiếng nói chuyện',
    'etc':              'Tạp âm khác',
    'silence':          'Yên lặng',
    'sneeze':           'Hắt hơi',
    'gasp':             'Thở gấp / Hớp không khí',
    'breathe':          'Tiếng thở',
    'sniffle':          'Sụt sịt / Hỉ mũi',
    'throat-clearing':  'Đằng hắng / Khạc họng',
}

# ── LogMel feature extractor (replicates utils/features.py) ───────────────────
# ── Pure Numpy Mel Filterbank (replaces librosa dependency) ────────────────────
def get_mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    m_min = hz_to_mel(fmin)
    m_max = hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    weights = np.zeros((n_mels, n_fft // 2 + 1))
    
    fdiff = np.diff(f_pts)
    ramps = np.subtract.outer(f_pts, fft_freqs)
    
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
        
    enorm = 2.0 / (f_pts[2:n_mels+2] - f_pts[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights

class LogMelExtractor:
    def __init__(self, sr, win_size, hop, n_mels):
        self.win_size = win_size
        self.hop = hop
        self.ham_win = np.hamming(win_size)
        self.melW = get_mel_filterbank(
            sr=sr, n_fft=win_size, n_mels=n_mels,
            fmin=20.0, fmax=sr / 2.0,
        ).T  # shape: (fft_bins+1, n_mels)

    def transform(self, audio: np.ndarray) -> np.ndarray:
        _, _, x = signal.spectral.spectrogram(
            audio,
            window=self.ham_win,
            nperseg=self.win_size,
            noverlap=self.hop,
            detrend=False,
            return_onesided=True,
            mode='magnitude',
        )
        x = x.T                        # (frames, fft_bins)
        x = np.dot(x, self.melW)       # (frames, n_mels)
        x = np.log(x + 1e-8)
        return x.astype(np.float32)


# ── Audio loading & padding ────────────────────────────────────────────────────
def load_audio(audio_path: str, target_sr: int, target_len: int) -> np.ndarray:
    """Load audio with soundfile, convert to mono, resample with scipy, pad/trim."""
    audio, sr = soundfile.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)   # stereo → mono
    audio = audio.astype(np.float32)
    if sr != target_sr:
        # scipy resample_poly — no numba dependency
        common = gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // common, sr // common)
    if len(audio) < target_len:
        n = math.ceil(target_len / len(audio))
        audio = np.tile(audio, n)
    return audio[:target_len].astype(np.float32)


# ── Model factory ──────────────────────────────────────────────────────────────
def build_model(backbone: str, n_classes: int):
    from pytorch.models import BaselineCnn, Vggish, ResNet, MobileNet
    backbone = backbone.lower()
    if 'baseline' in backbone:
        return BaselineCnn(n_classes)
    elif 'vgg' in backbone:
        return Vggish(n_classes)
    elif 'resnet' in backbone:
        return ResNet(n_classes)
    elif 'mobile' in backbone:
        return MobileNet(n_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. "
                         f"Choose from: baseline, vgg, resnet, mobilenet")


# ── Main prediction function ───────────────────────────────────────────────────
def predict(audio_path: str, model_path: str, backbone: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load & preprocess audio
    print(f"\n[INFO] Loading: {audio_path}")
    audio = load_audio(audio_path, target_sr=sample_rate,
                       target_len=audio_samples_flusense)
    print(f"       Sample rate : {sample_rate} Hz")
    print(f"       Duration    : {audio_samples_flusense / sample_rate:.2f} s  "
          f"(padded/trimmed to FluSense standard)")

    # 2. Extract LogMel features
    extractor = LogMelExtractor(
        sr=sample_rate, win_size=window_size,
        hop=overlap, n_mels=mel_bins,
    )
    feature = extractor.transform(audio)          # (frames, mel_bins)
    tensor = torch.tensor(feature).unsqueeze(0)  # (1, frames, mel_bins)
    tensor = tensor.to(device)

    # 3. Load model weights
    print(f"\n[INFO] Model     : {backbone} | n_classes = {classes_num_flusense}")
    print(f"       Weights   : {model_path}")

    model = build_model(backbone, classes_num_flusense)
    map_loc = torch.device(device)
    checkpoint = torch.load(model_path, map_location=map_loc, weights_only=False)

    # Support both raw state_dict and {'model': state_dict} format
    state_dict = checkpoint.get('model', checkpoint)
    # Strip 'module.' prefix if saved with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. Inference
    with torch.no_grad():
        logits = model(tensor)                    # (1, n_classes)
        probs = torch.softmax(logits, dim=1)[0]   # (n_classes,)

    probs_np = probs.cpu().numpy()

    # 5. Sort & display
    order = np.argsort(probs_np)[::-1]

    print("\n" + "=" * 62)
    print("  KẾT QUẢ PHÂN LOẠI ÂM THANH")
    print("=" * 62)
    print(f"  File : {os.path.basename(audio_path)}")
    print("-" * 62)
    print(f"  {'Hạng':<6} {'Nhãn (EN)':<20} {'Ý nghĩa (VI)':<26} {'Xác suất':>8}")
    print(f"  {'-'*5}  {'-'*19}  {'-'*25}  {'-'*8}")

    for rank, idx in enumerate(order):
        label = LABELS[idx] if idx < len(LABELS) else f'class_{idx}'
        label_vi = LABEL_VI.get(label, label)
        pct = probs_np[idx] * 100
        star = ' *' if rank == 0 else '  '
        print(f"  #{rank+1}{star}  {label:<20}  {label_vi:<26}  {pct:>7.2f}%")

    top_label = LABELS[order[0]] if order[0] < len(LABELS) else f'class_{order[0]}'
    top_pct = probs_np[order[0]] * 100
    print("=" * 62)
    print(f"\n  ->  Kết luận: [{top_label.upper()}] - {LABEL_VI.get(top_label, top_label)}"
          f"  ({top_pct:.1f}%)")
    print()

    return {LABELS[i]: float(probs_np[i]) for i in range(len(LABELS))}


# ── CLI entry-point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify a single audio file into 9 sound categories')
    parser.add_argument('--audio_file', type=str, required=True,
                        help='Path to input audio file (.wav/.flac/.mp3)')
    parser.add_argument('--model_path', type=str,
                        default='save_model/best_model.pth',
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--backbone', type=str, default='baseline',
                        choices=['baseline', 'vgg', 'resnet', 'mobilenet'],
                        help='Model backbone architecture')
    args = parser.parse_args()

    if not os.path.isfile(args.audio_file):
        print(f"[ERROR] Audio file not found: {args.audio_file}")
        sys.exit(1)
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)

    predict(
        audio_path=args.audio_file,
        model_path=args.model_path,
        backbone=args.backbone,
    )