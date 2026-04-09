# Giải thích chi tiết: Đếm hạt gạo bằng xử lý ảnh

> Tài liệu này dành cho người mới bắt đầu học xử lý ảnh. Mọi khái niệm sẽ được giải thích từ cơ bản.

---

## Ý tưởng tổng quát

Hãy tưởng tượng bạn nhìn một đĩa gạo và đếm từng hạt. Mắt bạn tự động làm 3 việc:
1. **Nhìn rõ** - bỏ qua bụi bẩn, ánh sáng lóe
2. **Phân biệt** - biết đâu là gạo, đâu là nền
3. **Đếm** - đếm từng hạt, kể cả hạt dính nhau

Chương trình của chúng ta cũng làm đúng 3 bước đó, nhưng bằng toán học.

---

## Bước 0: Ảnh là gì đối với máy tính?

Một ảnh xám (grayscale) là một **bảng số** (ma trận). Mỗi ô (pixel) chứa một số từ 0 (đen) đến 255 (trắng).

```
Ví dụ ảnh 4x4 pixel:
  120  130  125  128      ← nền xám
  130  220  210  132      ← có 2 pixel sáng (hạt gạo!)
  128  215  205  127      ← hạt gạo tiếp tục
  125  130  128  130      ← nền xám
```

Hạt gạo = pixel **sáng hơn** nền. Nhiệm vụ: tìm và đếm các **cụm pixel sáng**.

---

## Bước 1: Khử nhiễu - "Lau kính trước khi nhìn"

### Nhiễu là gì?
Nhiễu = những pixel bị sai giá trị, giống như vết bẩn trên kính.

### 1a. Nhiễu Salt & Pepper (Muối & Tiêu)
Một số pixel ngẫu nhiên bị chuyển thành trắng (muối=255) hoặc đen (tiêu=0).

```
Ảnh gốc:     Ảnh bị nhiễu:
120 130 125   120  0  125      ← pixel 130 bị thành 0 (tiêu)
130 200 132   130 200 255      ← pixel 132 bị thành 255 (muối)
```

**Cách xử lý: Median Filter (Lọc trung vị)**

Lấy một vùng nhỏ (ví dụ 5x5 pixel), sắp xếp giá trị, chọn **giá trị ở giữa**.

```
Vùng 3x3:  [0, 120, 125, 125, 128, 130, 130, 132, 200]
                              ↑
                        Trung vị = 128 (hợp lý!)
```

Vì pixel nhiễu (0 hoặc 255) luôn nằm ở đầu/cuối danh sách, nên trung vị sẽ bỏ qua chúng. Đây là lý do median filter xử lý Salt & Pepper **cực kỳ hiệu quả**.

### 1b. Nhiễu Sinusoidal (Sóng sin)
Ảnh bị phủ một lớp sọc sáng-tối xen kẽ (như rèm cửa), tạo ra bởi sóng sin.

```
Ảnh gốc:          Nhiễu sinus:       Ảnh bị nhiễu:
120 130 125   +    30  -30   30   =   150 100 155
130 200 132        30  -30   30       160 170 162
```

Sọc này **lặp đi lặp lại đều đặn** theo chiều ngang.

**Cách xử lý: Trừ pattern sinus**

Ý tưởng: Nếu sọc lặp đều theo cột, thì trung bình mỗi cột sẽ cho thấy pattern sinus.

```
Cột 1: trung bình = 150  (cao)
Cột 2: trung bình = 100  (thấp)
Cột 3: trung bình = 155  (cao)
→ Pattern sinus: [+25, -25, +25]
→ Trừ đi: ảnh sạch!
```

Cụ thể:
1. Tính trung bình từng cột → được 1 đường cong (chứa cả thông tin ảnh + sinus)
2. Dùng **median filter 1D** (cửa sổ lớn=101) → được đường baseline (chỉ thông tin ảnh)
3. Hiệu = sinus thuần túy → trừ khỏi ảnh gốc

### 1c. Gaussian Blur (Làm mịn)
Sau khi khử nhiễu chính, dùng Gaussian Blur để **làm mịn** ảnh thêm.

Gaussian Blur = thay mỗi pixel bằng **trung bình có trọng số** của vùng xung quanh, pixel gần hơn có trọng số lớn hơn (theo hình chuông Gauss).

```
Trọng số Gaussian 3x3:
  1  2  1
  2  4  2    ÷ 16
  1  2  1
```

Kết quả: ảnh mượt hơn, giảm nhiễu nhỏ còn sót, nhưng giữ được cạnh tương đối rõ.

---

## Bước 2: Phân đoạn - "Đâu là gạo, đâu là nền?"

### Threshold (Ngưỡng) là gì?
Chọn một giá trị ngưỡng T, rồi:
- Pixel > T → **trắng** (255) = gạo
- Pixel ≤ T → **đen** (0) = nền

```
Ảnh xám:          T = 150          Ảnh nhị phân:
120 200 130   →   đen trắng đen   →   0  255  0
210 125 190       trắng đen trắng     255  0  255
```

### Tại sao dùng Adaptive Threshold?

**Otsu** (ngưỡng toàn cục): Chọn 1 giá trị T duy nhất cho cả ảnh.
- Vấn đề: nếu nền không đều (góc trái sáng hơn góc phải), 1 ngưỡng không đủ.

**Adaptive Threshold** (ngưỡng cục bộ): Mỗi pixel có **ngưỡng riêng** = trung bình vùng xung quanh + hằng số C.

```
Vùng sáng (nền=180):  ngưỡng = 180 + C → gạo phải > 172
Vùng tối  (nền=100):  ngưỡng = 100 + C → gạo phải > 92
```

Như vậy dù nền sáng tối khác nhau, gạo vẫn được tách ra chính xác.

**Tham số:**
- `blockSize = 51`: kích thước vùng lân cận để tính trung bình (51x51 pixel, đủ lớn để chứa vài hạt gạo)
- `C = -8`: dịch ngưỡng xuống 8, giúp bắt được cả các hạt gạo không quá sáng

---

## Bước 3: Hậu xử lý hình thái học - "Sửa lỗi"

Sau threshold, ảnh nhị phân (trắng/đen) có thể có lỗi:
- Đốm trắng nhỏ = nhiễu (không phải gạo)
- Lỗ đen nhỏ = bên trong hạt gạo bị thiếu

### Opening = Erosion + Dilation (Xóa đốm nhỏ)

**Erosion (co)**: Thu nhỏ vùng trắng. Đốm nhỏ biến mất hoàn toàn.
```
Trước erosion:     Sau erosion:
. . . . .          . . . . .
. # . . .          . . . . .     ← đốm nhỏ mất
. . . . .          . . . . .
. # # # .          . . # . .     ← hạt gạo bị co lại
. # # # .          . . # . .
. # # # .          . . # . .
. . . . .          . . . . .
```

**Dilation (giãn)**: Phình vùng trắng ra. Phục hồi kích thước hạt gạo.
```
Sau dilation:
. . . . .
. . . . .          ← đốm nhỏ không quay lại (vì đã mất hoàn toàn)
. . . . .
. # # # .          ← hạt gạo trở lại kích thước gần đúng
. # # # .
. # # # .
. . . . .
```

→ **Opening = Erosion + Dilation** → giữ hạt gạo, xóa đốm nhiễu.

### Closing = Dilation + Erosion (Lấp lỗ nhỏ)
Ngược lại: phình ra trước (lấp lỗ nhỏ) rồi co lại (trả kích thước).

---

## Bước 4: Tách hạt dính - Watershed

### Vấn đề
Một số hạt gạo chạm nhau, tạo thành 1 blob trắng lớn. Nếu chỉ đếm blob → đếm thiếu.

```
2 hạt dính nhau = 1 blob:
. . . . . . . .
. # # # # # # .    ← Trông như 1 vật thể, nhưng thực ra là 2 hạt
. # # # # # # .
. . . . . . . .
```

### Ý tưởng Watershed (Đường phân thủy)

Hãy tưởng tượng ảnh nhị phân là **bản đồ địa hình**:

1. **Distance Transform**: Tính khoảng cách từ mỗi pixel trắng đến cạnh gần nhất.
   - Tâm hạt gạo = xa cạnh nhất = "đỉnh núi"
   - Cạnh hạt = gần cạnh nhất = "thung lũng"

```
Blob dính:          Distance Transform:
. . . . . . . .    . . . . . . . .
. # # # # # # .    . 1 2 2 1 1 2 .    ← 2 đỉnh = 2 hạt!
. # # # # # # .    . 1 2 2 1 1 2 .
. . . . . . . .    . . . . . . . .
```

2. **Tìm đỉnh**: Chọn các vùng có distance > ngưỡng → đây là **chắc chắn là tâm hạt**.

3. **Watershed**: Tưởng tượng "đổ nước" từ các đỉnh. Nước lan ra, khi 2 vùng nước gặp nhau → đó là **ranh giới** giữa 2 hạt.

---

## Bước 5: Đếm và lọc

Sau watershed, dùng `findContours` để tìm đường viền từng vùng, rồi tính diện tích.

### Lọc theo diện tích
- Tính **diện tích trung vị** (median) của tất cả contour
- Quá nhỏ (< 15% median): nhiễu → bỏ
- Bình thường (15% - 400% median): 1 hạt gạo → đếm 1
- Quá lớn (> 400% median): nhiều hạt dính → ước lượng = diện tích / median

```
Ví dụ: median = 500 pixel
- Contour 80 pixel  → < 75 → bỏ (nhiễu)
- Contour 450 pixel → bình thường → đếm 1
- Contour 1500 pixel → > 2000 không, 1500/500 = 3 → đếm 3 hạt
```

---

## Tóm tắt bằng hình ảnh

```
Ảnh gốc (có nhiễu)
    │
    ▼
[Khử nhiễu Sinus]     ← Trừ pattern sọc dọc (nếu có)
    │
    ▼
[Median Filter]        ← Xóa điểm muối/tiêu
    │
    ▼
[Gaussian Blur]        ← Làm mịn
    │
    ▼
[Adaptive Threshold]   ← Tách gạo (trắng) khỏi nền (đen)
    │
    ▼
[Opening]              ← Xóa đốm nhiễu nhỏ
    │
    ▼
[Closing]              ← Lấp lỗ nhỏ trong hạt
    │
    ▼
[Watershed]            ← Tách hạt dính nhau
    │
    ▼
[Đếm contour]         ← Đếm + lọc theo diện tích
    │
    ▼
Kết quả: 95 hạt gạo!
```

---

## Kết quả thực tế

| Ảnh | Loại nhiễu | Kết quả |
|-----|-----------|---------|
| `1_wIXlvBeAFtNVgJd49VObgQ.png` | Không có | 95 hạt |
| `...Salt_Pepper_Noise1.png` | Salt & Pepper | 95 hạt |
| `..._sinus.png` | Sinusoidal | 96 hạt |
| `1_zd6ypc20QAIFMzrbCmJRMg.png` | Ảnh tối khác | 80 hạt |

3 ảnh đầu (cùng gốc, khác nhiễu) cho kết quả **gần giống nhau** (95-96), chứng tỏ pipeline khử nhiễu hoạt động rất tốt.

---

## Câu hỏi thường gặp

**Q: Tại sao không dùng Otsu thay vì Adaptive Threshold?**
A: Otsu chọn 1 ngưỡng cho cả ảnh. Nếu nền không đều (sáng 1 góc, tối 1 góc), Otsu sẽ sai ở vùng tối hoặc vùng sáng. Adaptive tính ngưỡng riêng cho từng vùng nên chính xác hơn.

**Q: Tại sao dùng Median Filter thay vì Gaussian Blur để khử Salt & Pepper?**
A: Gaussian Blur tính trung bình → pixel nhiễu (0 hoặc 255) vẫn ảnh hưởng kết quả. Median chọn giá trị giữa → bỏ qua hoàn toàn pixel cực trị.

**Q: Tại sao cần Watershed? Đếm blob thẳng không được sao?**
A: Được, nhưng thiếu chính xác. Khi 2 hạt chạm nhau, chúng thành 1 blob → đếm thiếu. Watershed tách chúng ra.

**Q: Kernel size 5x5 nghĩa là gì?**
A: Khi xử lý mỗi pixel, ta nhìn vùng 5x5 pixel xung quanh nó (tức 25 pixel). Kernel lớn hơn = mịn hơn nhưng mất chi tiết hơn.
