# Báo cáo: Đếm hạt gạo bằng xử lý ảnh OpenCV

## 1. Mô tả bài toán
Đếm số lượng hạt gạo trong ảnh. Các ảnh có thể bị nhiễu bởi nhiều loại nhiễu khác nhau (Salt & Pepper, Sinusoidal). Yêu cầu: **một chuỗi xử lý duy nhất** áp dụng cho tất cả các ảnh.

## 2. Chuỗi xử lý ảnh

### Bước 1: Khử nhiễu Sinusoidal (Column-mean subtraction)
- **Phương pháp**: Tính trung bình cột (`col_means`), ước lượng baseline bằng median filter 1D (kernel=101), trừ thành phần sinus ra khỏi ảnh.
- **Tham số**: `medfilt(kernel_size=101)`, ngưỡng phát hiện `std > 3`
- **Lý do**: Nhiễu sinus tạo sọc dọc đều → trung bình theo cột sẽ bắt được pattern này. Median filter 1D tách baseline (thông tin ảnh) khỏi sinus (nhiễu tuần hoàn). Chỉ áp dụng khi phát hiện năng lượng sinus đáng kể (`std > 3`), nên không ảnh hưởng đến ảnh không bị nhiễu sinus.

### Bước 2: Khử nhiễu Salt & Pepper + Làm mịn
- **Phương pháp**: Median Blur → Gaussian Blur
- **Tham số**: `medianBlur(ksize=5)`, `GaussianBlur(ksize=5x5, sigma=1)`
- **Lý do**: Median filter rất hiệu quả cho nhiễu Salt & Pepper vì nó loại bỏ các pixel cực trị mà không làm mờ cạnh. Gaussian blur sau đó làm mịn thêm để giảm nhiễu còn sót.

### Bước 3: Phân đoạn (Adaptive Threshold)
- **Phương pháp**: `adaptiveThreshold` với Gaussian weighting
- **Tham số**: `blockSize=51`, `C=-8`
- **Lý do**: Otsu's threshold không phù hợp vì nền ảnh không đồng đều (gradient chiếu sáng, vết nhiễu sinus còn sót). Adaptive threshold tính ngưỡng cục bộ cho từng vùng, giúp phân đoạn chính xác ngay cả khi nền thay đổi. `blockSize=51` đủ lớn để bao phủ vài hạt gạo, `C=-8` điều chỉnh để tách rõ hạt gạo khỏi nền.

### Bước 4: Hậu xử lý hình thái học
- **Phương pháp**: Opening → Closing
- **Tham số**: Kernel ellipse 3x3, mỗi phép 2 iterations
- **Lý do**:
  - **Opening** (erosion → dilation): Loại bỏ các đốm nhiễu nhỏ còn sót sau threshold.
  - **Closing** (dilation → erosion): Lấp các lỗ nhỏ bên trong hạt gạo.

### Bước 5: Tách hạt dính bằng Watershed
- **Phương pháp**: Distance Transform → Watershed
- **Tham số**: `distanceTransform(DIST_L2)`, ngưỡng `0.4 * max`, dilate 3 iterations
- **Lý do**: Một số hạt gạo chạm nhau tạo thành blob lớn. Distance transform tìm tâm của từng hạt (vùng xa cạnh nhất), watershed dùng các tâm này làm marker để tách các hạt dính.

### Bước 6: Đếm và lọc theo diện tích
- **Phương pháp**: `findContours` → lọc theo diện tích → ước lượng blob lớn
- **Tham số**: `min_area = median * 0.15`, `max_area = median * 4.0`
- **Lý do**: Loại bỏ nhiễu nhỏ (< 15% diện tích trung vị) và ước lượng số hạt trong blob lớn (> 4x diện tích trung vị) bằng cách chia diện tích blob cho diện tích trung vị.

## 3. Kết quả

| Ảnh | Số hạt gạo |
|-----|-----------|
| Ảnh gốc | 95 |
| Salt & Pepper Noise | 95 |
| Sinusoidal Noise | 96 |
| Ảnh tối (khác) | 80 |

## 4. Phân tích kết quả

### Kết quả tốt
- **3 ảnh đầu** (cùng 1 ảnh gốc với các loại nhiễu khác nhau) cho kết quả **gần như giống nhau (95-96)**, chứng tỏ pipeline khử nhiễu hoạt động hiệu quả.
- **Salt & Pepper**: Median filter xử lý hoàn hảo, kết quả giống hệt ảnh gốc.
- **Sinusoidal**: Column-mean subtraction loại bỏ sọc dọc rất tốt, kết hợp adaptive threshold giúp phân đoạn chính xác.

### Hạn chế
- **Ảnh tối** (`1_zd6ypc20QAIFMzrbCmJRMg.png`): Đây là ảnh khác (gạo tối trên nền đen), contrast thấp hơn. Kết quả 80 hạt có thể chưa hoàn toàn chính xác do một số hạt có độ tương phản quá thấp với nền.
- **Watershed** đôi khi chia quá mức (over-segmentation) hoặc chưa tách hết các hạt dính, tùy thuộc vào ngưỡng distance transform.
- Tham số `blockSize` và `C` trong adaptive threshold được chọn thủ công dựa trên thực nghiệm, có thể chưa tối ưu cho mọi trường hợp.

## 5. Cách chạy

```bash
# Chạy trên 1 ảnh
python count_rice.py Proj1.2/ten_anh.png

# Chạy trên tất cả ảnh
python count_rice.py
```

## 6. Thư viện sử dụng
- OpenCV (`cv2`)
- NumPy
- SciPy (`scipy.signal.medfilt`)
