import sys
import argparse
from mobilenet_detector import MobileNetCoughDetector

def main():
    parser = argparse.ArgumentParser(description="Phân tích toàn bộ file âm thanh với mô hình MobileNet (dự đoán từng đoạn).")
    parser.add_argument('audio_file', type=str, help="Đường dẫn đến file âm thanh cần test (VD: audio.wav)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Ngưỡng xác suất để coi là tiếng ho (mặc định 0.5)")
    args = parser.parse_args()

    print(f"[*] Đang khởi tạo mô hình...")
    detector = MobileNetCoughDetector(threshold=args.threshold)
    
    print(f"[*] Đang nạp và phân tích file: {args.audio_file}")
    print(f"[*] Xin vui lòng chờ trong giây lát...")
    
    try:
        # Sử dụng API mới phân tích sliding window toàn file
        result = detector.analyze_file(args.audio_file)
        
        print("\n" + "="*40)
        print(" TỔNG QUAN FILE ÂM THANH")
        print("="*40)
        print(f"👉 File test                    : {args.audio_file}")
        print(f"👉 Chứa tiếng ho?               : {'CÓ' if result['has_cough'] else 'KHÔNG'} (Max prob: {result['max_cough_prob']:.4f})")
        print(f"👉 Các loại âm thanh phát hiện  : {', '.join(result['detected_classes'])}")
        
        print("\n" + "="*40)
        print(" CHI TIẾT TỪNG PHÂN ĐOẠN ÂM THANH (Dài 1 giây / bước dịch 0.5s)")
        print("="*40)
        
        # Sắp xếp output theo dòng thời gian
        for chunk in result['chunks']:
            start = chunk['start_time']
            end = chunk['end_time']
            p_class = chunk['predicted_class']
            cough_p = chunk['cough_probability']
            is_cough = "[CẢNH BÁO: HO]" if chunk['is_cough'] else ""
            
            print(f"[{start:05.1f}s - {end:05.1f}s] Dấu hiệu chính: {p_class:<15} | Tỷ lệ ho: {cough_p:.4f} {is_cough}")
            
    except FileNotFoundError:
        print(f"⚠️ Lỗi: Không tìm thấy file âm thanh '{args.audio_file}'")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Đã có lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
