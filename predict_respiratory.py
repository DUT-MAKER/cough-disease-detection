#!/usr/bin/env python
# coding: utf-8
"""
predict_respiratory.py
──────────────────────
Dự đoán bệnh hô hấp từ một file âm thanh tiếng ho.

Hệ thống sử dụng:
  • FRILL (TensorFlow Hub) để trích xuất 2048 đặc trưng âm thanh sâu
  • XGBoost đã được huấn luyện trước trên tập Sound-Dr + Augmented data

Các lớp bệnh hô hấp:
  0 – Healthy    (Khỏe mạnh)
  1 – Asthma     (Hen suyễn)
  2 – COPD       (Phổi tắc nghẽn mãn tính)
  3 – Pneumonia  (Viêm phổi)
  4 – COVID-19

Cách dùng:
  source .venv/bin/activate
  python predict_respiratory.py <đường_dẫn_file_wav>

Ví dụ:
  python predict_respiratory.py cough/bad_cough_2021-08-01T03:23:07.662Z.wav
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, pickle
import numpy as np

# ─── Cấu hình đường dẫn ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = SCRIPT_DIR
SOUNDDR_DIR  = os.path.join(PROJECT_DIR, 'Sound-Dr')
SOUNDDR_DATA = os.path.join(SOUNDDR_DIR, 'sounddr_data')

MODEL_PATH  = os.path.join(SOUNDDR_DATA, 'output', 'xgb_respiratory_model.pkl')
META_PATH   = os.path.join(SOUNDDR_DATA, 'output', 'xgb_respiratory_meta.pkl')

RESP_CLASSES = ['Healthy', 'Asthma', 'COPD', 'Pneumonia', 'COVID']
RESP_CLASSES_VI = {
    'Healthy':   '🟢 Khỏe mạnh',
    'Asthma':    '🟡 Hen suyễn (Asthma)',
    'COPD':      '🟠 Phổi tắc nghẽn (COPD)',
    'Pneumonia': '🔴 Viêm phổi (Pneumonia)',
    'COVID':     '🔴 Nghi ngờ COVID-19',
}

SR = 16000  # Sample rate chuẩn

# ─── Hàm trích xuất đặc trưng ────────────────────────────────────────────────
def extract_features_frill(wav_path: str, frill_dim: int) -> np.ndarray:
    """Trích xuất FRILL embedding từ file WAV."""
    sys.path.insert(0, os.path.abspath(SOUNDDR_DIR))
    try:
        from feature import make_nonsemantic_frill_nofrontend_feat
        feat = make_nonsemantic_frill_nofrontend_feat(wav_path)
        if feat.ndim == 1 and len(feat) != frill_dim:
            # Pad hoặc cắt nếu không đúng chiều
            vec = np.zeros(frill_dim, dtype=np.float32)
            vec[:min(len(feat), frill_dim)] = feat[:frill_dim]
            return vec
        return feat.astype(np.float32)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"Lỗi khi load feature.py: {e}")


def extract_features_mfcc(wav_path: str, frill_dim: int) -> np.ndarray:
    """MFCC fallback khi không có TensorFlow (chất lượng thấp hơn)."""
    import librosa
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=128)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])  # 256 chiều
    vec = np.zeros(frill_dim, dtype=np.float32)
    vec[:min(len(feat), frill_dim)] = feat[:frill_dim]
    return vec


# ─── Hàm dự đoán chính ───────────────────────────────────────────────────────
def predict(wav_path: str, verbose: bool = True, force_mfcc: bool = False) -> dict:
    """
    Dự đoán bệnh hô hấp từ file âm thanh tiếng ho.

    Args:
        wav_path: Đường dẫn tới file .wav
        verbose:  In kết quả ra màn hình
        force_mfcc: Ép buộc chạy luồng MFCC nhẹ (Bỏ qua cấu trúc siêu sâu XLA/FRILL)

    Returns:
        dict với keys: disease, disease_vi, confidence, all_probs
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Không tìm thấy file: {wav_path}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Chưa có model! Hãy chạy pytorch/respiratory_sounddr.py để train và lưu model trước.\n"
            f"Model cần ở: {MODEL_PATH}"
        )

    # Load model và meta
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)

    le            = meta['label_encoder']
    active_names  = meta['active_classes']  # Các class thực sự có trong train data
    frill_dim     = meta['frill_dim']

    if verbose:
        print("\n" + "="*55)
        print("  🏥 HỆ THỐNG CHẨN ĐOÁN BỆNH HÔ HẤP (TRACK B XGBoost)")
        print("="*55)
        print(f"  File âm thanh: {os.path.basename(wav_path)}")
        print(f"  Đang phân tích...")

    # Trích xuất đặc trưng
    try:
        # Ép CPU để tránh lỗi XLA JIT trên Local Venv GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if force_mfcc:
            raise Exception("Người dùng thiết lập cờ --mfcc thủ công")
        feat = extract_features_frill(wav_path, frill_dim)
        mode = "FRILL (Cao)"
    except Exception as e:
        if not force_mfcc:
            print(f"\n  [DEBUG Y TẾ] Không thể bật FRILL vì lỗi: {e}")
            mode = "MFCC fallback (Thấp – cần cài đặt môi trường chuẩn để dùng FRILL)"
        else:
            mode = "MFCC mode (Chạy thủ công nhanh, tiết kiệm RAM cứng)"
            
        feat = extract_features_mfcc(wav_path, frill_dim)

    # Dự đoán
    X = feat.reshape(1, -1)
    proba_encoded = model.predict_proba(X)[0]  # shape: (n_active_classes,)

    # Map xác suất về nhãn gốc
    all_probs = {name: 0.0 for name in RESP_CLASSES}
    for i, name in enumerate(active_names):
        all_probs[name] = float(proba_encoded[i])

    # Chẩn đoán chính (nhãn có xác suất cao nhất)
    pred_encoded  = int(np.argmax(proba_encoded))
    pred_disease  = active_names[pred_encoded]
    confidence    = float(proba_encoded[pred_encoded])

    if verbose:
        print(f"\n  ────────────────────────────────────────")
        print(f"  📊 Chất lượng đặc trưng: {mode}")
        print(f"\n  📋 Xác suất tất cả các bệnh:")
        for disease, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            bar   = "█" * int(prob * 20)
            space = "░" * (20 - len(bar))
            vi    = RESP_CLASSES_VI.get(disease, disease)
            print(f"    {vi:<32} {bar}{space} {prob*100:5.1f}%")

        print(f"\n  ════════════════════════════════════════")
        vi_result = RESP_CLASSES_VI.get(pred_disease, pred_disease)
        print(f"  ✅ KẾT QUẢ CHẨN ĐOÁN:  {vi_result}")
        print(f"     Mức độ tin cậy:       {confidence*100:.1f}%")

        if confidence < 0.5:
            print(f"\n  ⚠️  Tin cậy thấp (<50%). Vui lòng kiểm tra lại âm thanh.")
        print(f"  ════════════════════════════════════════\n")

    return {
        'disease':    pred_disease,
        'disease_vi': RESP_CLASSES_VI.get(pred_disease, pred_disease),
        'confidence': confidence,
        'all_probs':  all_probs,
    }


# ─── CLI Entry-point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nCách dùng:")
        print("  python predict_respiratory.py <đường_dẫn_file_wav>")
        print("\nVí dụ:")
        print("  python predict_respiratory.py cough/bad_cough_2021-08-01T03:23:07.662Z.wav")
        sys.exit(1)

    wav_input = sys.argv[1]

    # Hỗ trợ cả đường dẫn tương đối từ thư mục gốc project
    if not os.path.isabs(wav_input):
        wav_input = os.path.join(PROJECT_DIR, wav_input)

    try:
        result = predict(wav_input, verbose=True)
    except FileNotFoundError as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Lỗi không xác định: {e}")
        raise
