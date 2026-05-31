import sys
import numpy as np
import soundfile
from mobilenet_detector import MobileNetCoughDetector
import warnings
warnings.filterwarnings('ignore')

try:
    audio_file = sys.argv[1]
except IndexError:
    print("Vui lòng cung cấp file âm thanh.")
    sys.exit(1)

detector = MobileNetCoughDetector()

# 1. Thử đọc kiểu bình thường để phân tích giá trị trả về
print("================ DEBUG NGUYÊN BẢN (1 GIÂY ĐẦU) ================")
try:
    res = detector.predict_file(audio_file)
    print(res)
except Exception as e:
    print(e)
    
print("\n================ DEBUG PHÂN TÍCH TỪNG ĐOẠN ==================")
try:
    res_full = detector.analyze_file(audio_file)
    for chunk in res_full['chunks']:
        print(f"[{chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s] Lớp cao nhất: {chunk['predicted_class']:<10} | Xác suất ho: {chunk['cough_probability']:.4f}")
except Exception as e:
    print(e)
