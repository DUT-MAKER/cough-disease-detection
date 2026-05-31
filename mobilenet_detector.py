import os
import sys
import argparse
import math
import numpy as np
import soundfile
import torch
import torch.nn as nn
from scipy import signal
from scipy.signal import resample_poly
from math import gcd

# Cấu hình để import đúng các thư viện sibling
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    sample_rate, audio_samples_flusense,
    window_size, overlap, mel_bins,
    classes_num_flusense,
)
from pytorch.models import MobileNet

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
        ).T

    def transform(self, audio: np.ndarray) -> np.ndarray:
        res = signal.spectrogram(
            audio,
            window=self.ham_win,
            nperseg=self.win_size,
            noverlap=self.hop,
            detrend=False,
            return_onesided=True,
            mode='magnitude',
        )
        x = res[-1]
        x = x.T
        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        return x.astype(np.float32)

class MobileNetCoughDetector:
    def __init__(self, model_path='save_mobilenet/best_model.pth', threshold=0.5):
        self.model_path = os.path.join(PROJECT_ROOT, model_path)
        self.threshold = threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Thiết lập LogMel feature extractor chuẩn với config lúc train
        hop = window_size - overlap
        self.extractor = LogMelExtractor(
            sr=sample_rate,
            win_size=window_size,
            hop=hop,
            n_mels=mel_bins
        )
        
        # Khởi tạo model và tải dữ liệu từ checkpoint
        self.model = MobileNet(classes_num_flusense)
        self.model.to(self.device)
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()

    def _process_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        
        if sr != sample_rate:
            common = gcd(sample_rate, sr)
            audio = resample_poly(audio, sample_rate // common, sr // common)
            
        if len(audio) < audio_samples_flusense:
            n = math.ceil(audio_samples_flusense / len(audio))
            audio = np.tile(audio, n)
            
        audio = audio[:audio_samples_flusense].astype(np.float32)
        mel_spec = self.extractor.transform(audio)
        return mel_spec

    def predict_audio_array(self, audio: np.ndarray, sr: int) -> dict:
        mel_spec = self._process_audio(audio, sr)
        
        # Thêm batch size và channel dimension = 1
        x = torch.tensor(mel_spec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
        cough_idx = LABELS.index('cough')
        cough_prob = float(probs[cough_idx])
        
        return {
            'is_cough': cough_prob >= self.threshold,
            'cough_probability': cough_prob,
            'all_probabilities': {label: float(prob) for label, prob in zip(LABELS, probs)},
            'predicted_class': LABELS[np.argmax(probs)]
        }


    def predict_full_audio(self, audio: np.ndarray, sr: int, chunk_sec: float = 1.0, step_sec: float = 0.5):
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        
        if sr != sample_rate:
            common = gcd(sample_rate, sr)
            audio = resample_poly(audio, sample_rate // common, sr // common)
        
        chunk_len = int(chunk_sec * sample_rate)
        step_len = int(step_sec * sample_rate)
        
        if len(audio) < chunk_len:
            n = math.ceil(chunk_len / len(audio))
            padded_audio = np.tile(audio, n)[:chunk_len]
            res = self.predict_audio_array(padded_audio, sample_rate)
            res['start_time'] = 0.0
            res['end_time'] = len(audio) / sample_rate
            return [res]
        
        results = []
        for start_idx in range(0, len(audio) - chunk_len + 1, step_len):
            chunk = audio[start_idx:start_idx + chunk_len]
            res = self.predict_audio_array(chunk, sample_rate)
            res['start_time'] = start_idx / sample_rate
            res['end_time'] = (start_idx + chunk_len) / sample_rate
            results.append(res)
            
        return results

    def analyze_file(self, file_path: str):
        audio, sr = soundfile.read(file_path)
        chunks_result = self.predict_full_audio(audio, sr)
        
        has_cough = any(c['is_cough'] for c in chunks_result)
        max_cough_prob = max((c['cough_probability'] for c in chunks_result), default=0.0)
        
        # Nhóm cảnh báo theo thời gian thực
        all_detected = {}
        for c in chunks_result:
            pred = c['predicted_class']
            if pred not in all_detected:
                all_detected[pred] = []
            all_detected[pred].append((c['start_time'], c['end_time'], c['cough_probability']))
            
        return {
            'has_cough': has_cough,
            'max_cough_prob': max_cough_prob,
            'detected_classes': list(all_detected.keys()),
            'timeline': all_detected,
            'chunks': chunks_result
        }
    def predict_file(self, file_path: str) -> dict:
        audio, sr = soundfile.read(file_path)
        return self.predict_audio_array(audio, sr)

if __name__ == '__main__':
    print("[*] Đang khởi tạo module MobileNetCoughDetector...")
    try:
        detector = MobileNetCoughDetector()
        print("[*] Trạng thái: Tải model thành công!")
        
        print(f"\n[Test] Sinh dữ liệu âm thanh ngẫu nhiên (1s - {sample_rate}Hz) để test...")
        # Tạo dữ liệu ngẫu nhiên thay thế cho microphone feed
        test_sr = sample_rate
        audio_data = np.random.randn(test_sr).astype(np.float32)
        
        print("[Test] Đang thực hiện dự đoán đưa vào model...")
        result = detector.predict_audio_array(audio_data, sr=test_sr)
        
        print("\n=== KẾT QUẢ DỰ ĐOÁN ===")
        print(f"👉 Có phải tiếng ho không? : {result['is_cough']}")
        print(f"👉 Xác suất tiếng ho       : {result['cough_probability']:.4f}")
        print(f"👉 Lớp dự đoán có khả năng : {result['predicted_class']}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Lỗi: {e}")
