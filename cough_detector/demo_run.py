#!/usr/bin/env python
# coding: utf-8
"""
demo_run.py
───────────
Chạy thử nghiệm Hệ thống 1 (Cough Detection).
Script này đóng vai trò như Bức tường lửa đầu tiên: Vàng lọt, Cát ở lại.

Cách dùng:
  source .venv/bin/activate
  python cough_detector/demo_run.py <đường_dẫn_file_âm_thanh>

Luồng thực tế:
  1. System 1 kiểm tra xem có tiếng ho nào không (YAMNet)
  2. Nếu có tiếng ho -> Đưa sang System 2 dự đoán bệnh (XGBoost)
"""

import os
import sys

# Đảm bảo có thể import các module ngoài
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cough_detector.detector import YAMNetCoughDetector

# Thử import System 2 (predict_respiratory.py) nếu có
try:
    from predict_respiratory import predict as predict_disease
    has_system_2 = True
except ImportError:
    has_system_2 = False

def detect_and_route(audio_path, force_mfcc=False):
    print("="*60)
    print(" 🏥 KHỞI ĐỘNG HỆ THỐNG GÁC CỔNG (SYSTEM 1 - COUGH DETECTION)")
    print("="*60)
    
    print(f"[1] Phân tích Âm thanh: '{os.path.basename(audio_path)}'")
    
    # Khởi tạo detector với ngưỡng cẩn thận (0.25 - 0.40 là hợp lý)
    detector = YAMNetCoughDetector(threshold=0.30)
    
    # ─── BƯỚC 1: Lọc nhiễu / Phát hiện ho ────────────────────────────────
    print("[2] YAMNet đang quyét từng khung hình 0.96 giây...")
    result = detector.analyze(audio_path)
    
    is_cough = result['is_cough']
    confidence = result['max_confidence']
    timestamps = result['timestamps_sec']
    
    print(f"\n  📊 Kết quả lọc:")
    print(f"     - Độ tự tin (Cough Confidence): {confidence * 100:.1f}%")
    
    if is_cough:
        print(f"     - ✅ KẾT LUẬN: ĐÂY LÀ TIẾNG HO!")
        print(f"     - ⏱️ Xuất hiện rõ nhất ở các giây: {timestamps}")
        
        # ─── BƯỚC 2: Điều hướng sang System 2 ────────────────────────────
        print("\n" + "-"*60)
        print(" 🚀 ĐIỀU HƯỚNG SANG HỆ THỐNG 2 (SYSTEM 2 - CHẨN ĐOÁN BỆNH)")
        print("-"*60)
        
        if has_system_2:
            print("[3] Phân tích lâm sàng tiếng ho...\n")
            predict_disease(audio_path, verbose=True, force_mfcc=force_mfcc)
        else:
            print("⚠️ System 2 (predict_respiratory.py) chưa khả dụng.")
            
    else:
        print(f"     - ❌ KẾT LUẬN: KHÔNG PHÁT HIỆN TIẾNG HO!")
        print("\n⚠️ HỆ THỐNG BÁO LỖI:")
        print("   Vui lòng thử lại. Đảm bảo bạn ho thẳng vào micro.")
        print("   Hệ thống không tiếp nhận tiếng nói chuyện, vỗ tay hay tiếng ồn.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\nCách dùng:")
        print("  python cough_detector/demo_run.py <đường_dẫn_file_wav> [--mfcc]")
        print("\nVí dụ:")
        print("  python cough_detector/demo_run.py cough/bad_cough...wav")
        print("  python cough_detector/demo_run.py cough/bad_cough...wav --mfcc")
        sys.exit(1)

    wav_input = sys.argv[1]
    
    if not os.path.exists(wav_input):
        print(f"\n❌ Lỗi: Không tìm thấy file gốc: {wav_input}")
        sys.exit(1)

    # Chạy
    detect_and_route(wav_input, force_mfcc="--mfcc" in sys.argv)
