# Xây Dựng Hệ Thống Chẩn Đoán Bệnh Hô Hấp Từ Tiếng Ho (End-to-End Hybrid Pipeline)

Dự án Machine Learning cấp y tế (Medical ML) sử dụng kiến trúc tiên tiến nhằm tự động phát hiện, lọc nhiễu và chẩn đoán đa trạng thái bệnh lý hô hấp thông qua tiếng ho. Hệ thống vừa được nâng cấp toàn diện với **Custom MobileNet Cough Detection Pipeline**.

---

## 🏗️ 1. Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo luồng rẽ nhánh nối tiếp (Pipeline) gồm 2 giai đoạn:

### Giai đoạn 1: Hệ thống Phát Hiện Tiếng Ho (Cough Detection - Custom MobileNet)
*   **Mục tiêu:** Xác định đoạn âm thanh đầu vào có chứa tiếng ho hay không, bóc tách chính xác vị trí thời gian bị nhiễu bởi tạp âm (tiếng nói, hắt hơi, v.v.).
*   **Mô hình:** **Custom MobileNet** (Tối ưu riêng cho dữ liệu Audio 1 kênh).
*   **5 Cải tiến cốt lõi (Improved Pipeline):**
    1. **Timeline-Aware Audio Scanning:** Khắc phục nhược điểm đoán trên 1 đoạn tĩnh nguyên file. Hệ thống tự động chia audio thành các cửa sổ nhỏ `1.0s` và trượt (hop size) `0.5s`, giúp bắt chính xác các đoạn ho cực ngắn.
    2. **Audio Feature Standardization:** Âm thanh đầu vào được tự động chuẩn hóa khắt khe: chuyển đổi Mono -> Resample đồng bộ tần số lấy mẫu (16 kHz) -> Cắt/pad độ dài -> Sinh đặc trưng Log-Mel Spectrogram ổn định.
    3. **Custom MobileNet cho Audio:** Kiến trúc nhẹ dùng Depthwise Separable Convolution, nhận đầu vào `(1, 64)`, đi qua các block MobileNet và kết thúc bằng `Global Max Pooling` cùng `Dropout (0.15)`. Đảm bảo tốc độ quét realtime mà không làm mất "vân âm thanh".
    4. **Improved Head & Decision Logic:** Đầu ra dự đoán trên phân phối xác suất 9 lớp âm thanh khác nhau (cough, speech, sneeze, v.v.), sau đó trích xuất độc lập probability của lớp `cough` để áp dụng ngưỡng quyết định linh hoạt.
    5. **Temporal Aggregation:** Kết quả quét không chỉ là "Có/Không", mà là báo cáo timeline chi tiết danh sách các mốc thời gian xuất hiện tiếng ho trong file chạy.

### Giai đoạn 2: Hệ thống Chẩn Đoán Lai (Hybrid Disease Classification - "The Clinician")
Tiếp nhận đầu ra "tiếng ho sạch" đã được bóc tách và định vị từ Giai đoạn 1, hệ thống chuyển sang phân tích chuyên sâu nhằm chẩn đoán xem tiếng ho đó thuộc về bệnh lý nào. Để đạt độ chính xác chuẩn y tế, chúng tôi thiết kế kiến trúc phân loại **Hybrid Feature Fusion (ResNet + FRILL + XGBoost)**, kết hợp cả góc nhìn "hình ảnh" và "âm thanh" của một đặc điểm ho.

*   **Trích xuất Đặc trưng Hình ảnh Phổ (Visual-based Features):** 
    *   Âm thanh được biến đổi thành dạng phổ Log-Mel Spectrogram (64 mel bins, max frames 626).
    *   Sử dụng mạng lõi **ResNet (Residual Network)** đã pre-train trên đặc trưng âm thanh như một bộ trích xuất đặc trưng (Feature Extractor). Trọng số từ layer Linear phân loại cuối cùng được tinh chỉnh bỏ đi, thu về vector đặc trưng tinh gọn ($256$ dimensions) đại diện cho góc nhìn cấu trúc "vân ho" trên biểu đồ phổ.
*   **Trích xuất Đặc trưng Âm học Sâu (Acoustic-based Features):** 
    *   Sử dụng mô hình **FRILL** (từ TensorFlow Hub - Non-semantic Speech Representations). Xử lý trực tiếp tín hiệu âm thanh thô để lấy ra các biểu diễn không mang tính ngữ nghĩa (non-semantic biểu hiện của độ khàn, rung, luồng hơi thở, v.v...) với vector đầu ra khổng lồ ($4096$ dimensions).
*   **Feature Fusion & Phân loại bằng XGBoost (Classification):** 
    *   Ghép nối hai tập đặc trưng khác hệ (CNN + FRILL), tạo ra siêu vector chiếu sâu.
    *   Huấn luyện bộ phân loại **XGBoost Classifier** tốc độ cao (chạy thẳng trên GPU qua `tree_method='hist'`, `device='cuda'`) với cấu hình tối ưu để đánh giá xác suất mắc bệnh.
    *   **Xử lý mất cân bằng lớp:** Hệ thống tự động thiết lập thuộc tính `scale_pos_weight` động dựa trên tỉ lệ dữ liệu thực tế (VD: Tỉ lệ ho do COVID-19 trong tự nhiên rất thấp).
    *   **Khuôn khổ Đánh giá Lâm sàng:** Đạt tiêu chuẩn khắt khe với **5-Fold Stratified Cross-Validation**, đảm bảo AUC và các metric báo cáo không gặp tình trạng quá khớp (overfitting).

Hệ thống hiện tại tập trung giải quyết bài toán chẩn đoán cốt lõi:
*   **Chẩn đoán phân biệt chuyên sâu (Differential Diagnosis):** Phân biệt ho do Bệnh lý khác (Symptomatic) vs. COVID-19 (`hybrid_coughvid_symptomatic.py`).

---

## 📊 2. Dữ Liệu & Học Chuyển Giao (Dataset & Transfer Learning)

Để đảm bảo hiệu suất tốt nhất trên tập dữ liệu y tế khan hiếm, dự án ứng dụng chiến lược **Học chuyển giao (Transfer Learning)** xuyên suốt cả 2 giai đoạn, đi kèm với quy trình thiết kế dữ liệu đa tầng.

### Giai đoạn 1: Hệ thống Phát Hiện Tiếng Ho (Cough Detection)
*   **Bộ dữ liệu đa âm thanh (9 Classes):** Mô hình được huấn luyện trên một tập hợp các âm thanh môi trường và con người gồm 9 lớp (ho, tiếng nói, hắt hơi, cười, im lặng, v.v.). Sự phân bố thực tế đang mất cân đối nghiêm trọng khi lớp `Cough` (Tiếng ho) chỉ chiếm ~15.7%.
*   **Kỹ thuật Tăng cường mụn tiêu (Targeted Augmentation):** Nhằm bù đắp sự thiếu hụt, hệ thống nội suy và sinh dữ liệu nhân tạo trực tiếp cho lớp mục tiêu `cough` (thêm nhiễu, thay đổi cường độ, dịch thời gian). Các mẫu này được lưu tại `cough/augmented` và đồng bộ qua `data_augmented.csv`, giúp mô hình phân định ranh giới (decision boundary) chính xác hơn.
*   **Transfer Learning (Custom MobileNet):** Thay vì train từ đầu (scratch), chúng tôi tận dụng lợi thế trích xuất đặc trưng hình ảnh của CNN thông qua chuẩn hóa âm thanh sang dạng ảnh Log-Mel Spectrogram. Cấu trúc Custom MobileNet với các block `Depthwise Separable Conv` được tinh chỉnh (fine-tuning) lại Head layer để thích ứng với 9 phân lớp âm thanh, cho độ nhạy cực cao kể cả với thời lượng rất ngắn (1 giây).

### Giai đoạn 2: Hệ thống Chẩn Đoán Cấp Lâm Sàng (Disease Classification)
*   **Bộ dữ liệu khổng lồ COUGHVID (EPFL):** Chuyển sang Giai đoạn 2, chúng tôi tận dụng kho âm thanh ho cộng đồng lớn nhất thế giới. Ở đây, dữ liệu nhiễu đã được lọc và chuẩn hóa lại qua metadata `coughvid_covid_cough_vs_other_cough.csv`. Yêu cầu là phân tách những bệnh nhân Ho có triệu chứng đường hô hấp (Symptomatic) và Ho do mắc COVID-19.
*   **Deep Feature Extraction (Frozen Transfer Learning):** Không dùng End-to-End CNN thông thường do độ phức tạp y tế, chúng tôi áp dụng Transfer Learning ở dạng "đóng băng" (Frozen Feature Extraction) kết hợp từ 2 siêu mô hình:
    1.  **Chuyển giao hình ảnh phổ (ResNet Backbone):** Sử dụng mạng ResNet đã được hội tụ trên dữ liệu âm học phổ cứng. Cắt bỏ lớp phân loại quyết định (Classification `fc1`), chỉ dùng phần gốc của mạng tóm gọn đặc trưng chuỗi thời gian nén vào vector `256` chiều - đóng vai trò như một phân tích viên "đọc hình ảnh X-Quang thanh quản".
    2.  **Chuyển giao âm học sâu (FRILL - Google):** Kế thừa tri thức từ mô hình mô tả giọng nói phi ngữ nghĩa (Non-semantic Speech) của Google, FRILL xử lý âm thanh thô để "nghe" được độ rung, độ đặc, tiếng rít luồng hơi, xuất ra vector `4096` chiều. Đây là dạng Transfer Learning trực tiếp mang lại giá trị cao nhất cho chẩn đoán y tế âm thanh. 
*   **Feature Fusion & Machine Learning:** Nẹp 2 khối tri thức (256 + 4096) thành một chiều suy luận chung, huấn luyện trên XGBoost có gán trọng số mất cân bằng (`scale_pos_weight`).

---

## 🚀 3. Hướng Dẫn Huấn Luyện & Khởi Chạy (Usage)

### Cài đặt môi trường
Dự án sử dụng `uv` (hoặc `pip`) giúp cài đặt môi trường sạch và quản lý dependencies:
```bash
# Cài đặt nền tảng
uv pip install torch numpy scipy soundfile librosa scikit-learn
```

### Quy trình Huấn Luyện Giai Đoạn 1 (Detection Pipeline)
Đã được cấu hình tự động nhận diện và sử dụng tập `data_augmented.csv` nếu có.
```bash
# Train model quét ho với kiến trúc custom MobileNet
uv run pytorch/main.py train --respiratory --backbone mobilenet --save_dir save_mobilenet
```

### Chạy Dự Đoán Tương Tác Cắt Tiếng Ho (Inference - Giai Đoạn 1)
Chạy bộ máy dò tiếng ho bằng script detect với pipeline quét dòng thời gian tích hợp sẵn mô hình tốt nhất:
```bash
python mobilenet_detector.py
```

### Quy trình Huấn Luyện Giai Đoạn 2 (Hybrid Classification Pipeline)
Hệ thống sẽ chạy qua quy trình: Tự động trích xuất FRILL, trích xuất ResNet, lưu Cache để chạy nhanh trong tương lai, và train XGBoost bằng 5-Fold Stratified CV.

```bash
# Pipeline chẩn đoán phân biệt: Nhóm Có Triệu Chứng (Symptomatic - bệnh hô hấp khác) vs. COVID-19
uv run python pytorch/hybrid_coughvid_symptomatic.py
```

---

## 🏆 4. Kết Quả Đánh Giá (Evaluation Metrics)

Hệ thống MobileNet Binary (Tiếng ho vs. Còn lại) hiện tại trên bộ test nội bộ đạt hiệu năng đo lường:
*   **AUC:** ~0.958
*   **UAR:** ~0.736

---

## 📚 5. Nguồn Tham Khảo (References)
1. **MobileNet (Howard et al.):** Mạng thần kinh tích chập nhẹ cho thiết bị di động và biên.
2. **COUGHVID Dataset:** Ngân hàng dữ liệu tiếng ho lớn nhất thế giới từ EPFL.
