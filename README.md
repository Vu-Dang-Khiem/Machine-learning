# Fake News Detection using Machine Learning

## 1. Giới thiệu đề tài
Fake news (tin giả) là những thông tin sai sự thật, bịa đặt hoặc gây hiểu lầm, được cố tình lan truyền qua các phương tiện truyền thông hoặc mạng xã hội nhằm mục đích gây giật gân, thao túng dư luận, thu lợi kinh tế, hoặc phục vụ mục đích chính trị/xấu xa, bắt chước hình thức tin tức chính thống nhưng không được kiểm chứng.  
Đề tài này tập trung xây dựng hệ thống phát hiện tin giả tiếng Việt dựa trên các thuật toán Machine Learning.

**Mục tiêu:**
- Tiền xử lý dữ liệu văn bản tiếng Việt
- Huấn luyện mô hình phân loại tin thật / tin giả
- Đánh giá hiệu quả mô hình
- Xây dựng demo dự đoán tin mới (inference)
---

## 2. Dataset
Gồm 1.181 bài báo đã được gán nhãn:
Label = 0: Tin thật (Real News)
Label  = 1: Tin giả (Fake News)
Dataset đã được chia sẵn thành 3 tập khi thu thập dữ liệu:
    Training set (849 bài báo)
    Validation set (171 bài báo)
    Test set (161 bài báo)
- **Nguồn dữ liệu:**  
  Dataset tin giả tiếng Việt được tổng hợp từ các nguồn báo chí và mạng xã hội.
- **Link tải dataset:**  
https://www.kaggle.com/datasets/goumanguyen/vietnamese-fake-news-dataset-pbl7

- **Mô tả các cột:**

| Tên cột  | Mô tả                   |
|----------|-------------------------|
| label    | Nhãn (0: Fake, 1: Real) |
| Maintext | Nội dung bài báo        |

---

## 3. Pipeline xử lý
Quy trình thực hiện gồm các bước:

**Bước 1: Tiền xử lý văn bản**
Văn bản đầu vào được làm sạch và chuẩn hóa trước khi đưa vào mô hình, bao gồm các bước:
- Chuyển toàn bộ văn bản về chữ thường
- Loại bỏ URL, email, chữ số và các ký tự đặc biệt bằng biểu thức chính quy (regex)
- Chuẩn hóa khoảng trắng trong văn bản
- Tách từ tiếng Việt bằng thư viện Underthesea
**Bước 2: Biểu diễn văn bản**
Sử dụng phương pháp TF-IDF (Term Frequency – Inverse Document Frequency) để chuyển văn bản thành vector đặc trưng
Áp dụng n-gram (1, 2):
- Unigram (1-gram): biểu diễn thông tin từ vựng cơ bản
- Bigram (2-gram): giúp mô hình học được cách kết hợp từ và phong cách diễn đạt
**Bước 3: Chia dữ liệu**
Dataset được lấy từ Kaggle và đã được chia sẵn thành:
Tập huấn luyện (Train set)
Tập validation (Validation set)
Tập kiểm thử (Test set)
**Bước 4: Huấn luyện mô hình**
Huấn luyện và so sánh các mô hình phân loại văn bản:
- Naive Bayes
- Logistic Regression
- Linear SVM (LinearSVC)
Lựa chọn mô hình có hiệu quả tốt nhất để sử dụng cho bài toán phát hiện tin giả
**Bước 5: Đánh giá mô hình**
Đánh giá hiệu quả mô hình trên tập test bằng các chỉ số:
- Accuracy
- Precision
- Recall
- F1-score
Trực quan hóa kết quả bằng Confusion Matrix
**Bước 6: Inference (Dự đoán)**
Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho văn bản mới
Triển khai demo dự đoán thông qua:
- Notebook demo
- Ứng dụng Streamlit

---

## 4. Mô hình sử dụng
Các mô hình được thử nghiệm:
- Naive Bayes
- Logistic Regression
- Linear SVM (LinearSVC)

**Mô hình được chọn:** Linear SVM  
**Lý do:**
- Phù hợp với dữ liệu văn bản chiều cao
- Hiệu quả tốt với TF-IDF
- Thời gian huấn luyện nhanh

---

## 5. Kết quả
Mô hình đạt được các kết quả sau trên tập test:

- Accuracy: 94,4%
- Precision: 88,75%
- Recall: 100%
- F1-score: 94,03%

Confusion Matrix cho thấy mô hình phân biệt tốt giữa tin thật và tin giả,
Mô hình có xu hướng **ưu tiên giảm bỏ sót tin giả**, chấp nhận đánh nhầm một số tin thật.
.

---

## 6. Hướng dẫn chạy dự án
### 6.1 Cài môi trường
yêu cầu: **Python 3.11**
Cài các thư viện bằng lệnh
```bash
pip install -r requirements.txt
```
### 6.2 Chạy train 
```bash
python app/train.py
```
### 6.3 Chạy demo / inference
Cách 1: Chạy notebook demo
- Mở file demo/demo.ipynb
Sau đó chạy toàn bộ notebook để thử dự đoán tin mới.
Cách 2: Chạy ứng dụng Streamlit
- streamlit run app/app.py
Sau khi chạy, truy cập địa chỉ hiển thị trên terminal (thường là http://localhost:8501) để sử dụng giao diện dự đoán.

### 7. Cấu trúc thư mục dự án
FakeNewsDetection/
│
├── app/              # Source code chính (train, preprocess, predict, app.py)
├── demo/             # Notebook hoặc script demo inference
├── data/             # Data mẫu nhỏ hoặc README hướng dẫn tải data
├── models/           # Model đã train (.pkl)
├── reports/          # Báo cáo PDF/DOCX
├── slides/           # Slide thuyết trình PPTX/PDF
├── requirements.txt  # Danh sách thư viện
├── README.md         # Hướng dẫn dự án
└── .gitignore        # File loại trừ khi upload GitHub

### 8. Tác giả
Họ tên: Vũ Đăng Khiêm

Mã sinh viên: 12423017

Lớp: 124231