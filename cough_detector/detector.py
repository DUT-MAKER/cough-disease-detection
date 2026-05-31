"""
Lớp nhận diện tiếng ho sử dụng mô hình YAMNet (TensorFlow Hub).
Nhẹ, nhanh, và chạy trực tiếp trên CPU.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU to avoid libdevice/XLA JIT errors on local venv

import io
import csv
import urllib.request
import numpy as np
import librosa

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as e:
    import traceback
    traceback.print_exc()
    tf = None

YAMNET_URL = 'https://tfhub.dev/google/yamnet/1'

class YAMNetCoughDetector:
    def __init__(self, threshold=0.25):
        """
        Khởi tạo bộ lọc.
        :param threshold: Ngưỡng tự tin tối thiểu (0.0 -> 1.0) để quyết định là tiếng ho.
        """
        if tf is None:
            raise ImportError("Vui lòng cài tensorflow và tensorflow-hub: pip install tensorflow tensorflow-hub")
            
        self.threshold = threshold
        self.model = None
        self.class_names = []
        self.cough_indices = []
        self._load_model()

    def _load_model(self):
        print("[INFO] Đang tải mô hình YAMNet (chỉ lẩn đầu)...")
        # Load the YAMNet model from TF Hub
        self.model = hub.load(YAMNET_URL)
        
        # Lấy file map nhãn (YAMNet trả về 521 nhãn, mình cần biết index nào là nhãn ho)
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.class_names.append(row['display_name'])
                
        # Tìm các index liên quan đến việc ho
        target_classes = ['Cough', 'Throat clearing', 'Sneeze'] # Mở rộng nhẹ để không lọt
        for idx, name in enumerate(self.class_names):
            if name in target_classes:
                self.cough_indices.append(idx)
                
        print(f"[INFO] YAMNet tải thành công! Label ho nằm ở index: {self.cough_indices}")

    def analyze(self, wav_path):
        """
        Kiểm tra 1 file âm thanh có chứa tiếng ho hay không.
        :return: dict chứa kết quả boolean, điểm số cao nhất, và thời điểm.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Không tìm file: {wav_path}")

        # YAMNet yêu cầu đầu vào là mảng 1D, frame_rate = 16000Hz, giá trị [-1.0, 1.0]
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        waveform = y.astype(np.float32)

        # Trích xuất predictions (shape: [num_frames, 521])
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy() # Convert EagerTensor to numpy array

        # Lọc ra điểm số của riêng các nhãn ho ở mỗi frame (0.96 giây)
        cough_scores = scores_np[:, self.cough_indices]
        # Điểm ho cao nhất ở mỗi frame
        max_cough_per_frame = np.max(cough_scores, axis=1)
        
        # Chỉ số lớn nhất trên toàn bộ đoạn âm thanh
        global_max_score = np.max(max_cough_per_frame)
        
        # Nếu điểm cao nhất qua ngưỡng -> Có tiếng ho
        is_cough = bool(global_max_score >= self.threshold)
        
        # Tìm những thời điểm (số giây) mà tiếng ho xuất hiện rực rỡ nhất
        # YAMNet nhảy frame mỗi 0.48s
        frame_step = 0.48
        timestamps = []
        for frame_idx, score in enumerate(max_cough_per_frame):
            if score >= self.threshold:
                time_s = round(frame_idx * frame_step, 2)
                timestamps.append(time_s)

        return {
            'is_cough': is_cough,
            'max_confidence': float(global_max_score),
            'timestamps_sec': timestamps
        }
