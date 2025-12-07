# 🧠 Hand-drawn Digit Recognition AI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/Machine%20Learning-Softmax-orange)

Một ứng dụng web giúp nhận diện chữ số viết tay (0-9) sử dụng Machine Learning (Softmax Regression) được xây dựng từ con số 0 bằng NumPy.

## ✨ Tính năng nổi bật
* ✍️ **Vẽ trực tiếp:** Hỗ trợ vẽ mượt mà trên cả máy tính và điện thoại.
* 🎨 **Giao diện Tắc Kè Hoa:** Tự động đổi Theme màu sắc (Pastel/Dark/Teal) dựa theo Model được chọn.
* 🧠 **Multi-Model AI:** Tích hợp 3 mô hình xử lý khác nhau:
    1.  **Pixel Model:** Dựa trên độ đậm nhạt pixel gốc.
    2.  **Sobel Model:** Sử dụng thuật toán phát hiện cạnh (Edge Detection).
    3.  **Block Avg Model:** Nén ảnh để tăng tốc độ xử lý.
* 📊 **Biểu đồ trực quan:** Hiển thị xác suất dự đoán cho từng số.

## 📸 Demo
*(Cậu hãy chụp màn hình web lúc đang chạy, upload lên đây để người ta thấy độ đẹp nhé)*
![Demo Screenshot](https://via.placeholder.com/800x400?text=Place+Your+Screenshot+Here)

## 🛠️ Cài đặt và Chạy thử

1.  **Clone dự án:**
    ```bash
    git clone [https://github.com/USERNAME/REPO-NAME.git](https://github.com/USERNAME/REPO-NAME.git)
    cd REPO-NAME
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình API (Quan trọng):**
    * Mở file `frontend/script.js`.
    * Tìm hàm `predict()`.
    * Nếu muốn chạy local: Sửa link fetch thành `http://127.0.0.1:5000/predict`.
    * Nếu muốn dùng bản online: Giữ nguyên link Render.

4.  **Chạy Backend:**
    ```bash
    python backend/app.py
    ```

5.  **Mở Frontend:**
    Mở file `frontend/index.html` trên trình duyệt và trải nghiệm!
    
## 🤖 Cấu trúc thư mục
* `app.py`: Flask Backend xử lý ảnh và chạy model.
* `script.js`: Logic vẽ Canvas và gọi API.
* `style.css`: Định nghĩa các Theme màu sắc.
* `*.npz`: Các file trọng số model đã được huấn luyện.

## 🤝 Credits
Dự án được thực hiện bởi **TLUAT KHÙN**.
