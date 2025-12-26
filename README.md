# Traffic Analysis Solution - Vehicle Tracking & Wrong-way Detection
---
## 1. Project Overview (Tổng quan)

Giải pháp phân tích giao thông dựa trên CCTV, tập trung vào các tính năng tự động hóa giám sát và phân tích hành vi giao thông:

- **Detection & Tracking:** Sử dụng **YOLO11** kết hợp **ByteTrack** (state-of-the-art) để phát hiện và theo dõi phương tiện với độ chính xác cao.
- **Wrong-way Detection:** Phát hiện xe đi ngược chiều dựa trên cấu hình Polygon Zones và hướng di chuyển dự kiến.
- **Speed Estimation:** Ước lượng vận tốc (km/h) sử dụng kỹ thuật Perspective Transformation (homography).
- **Multi-stream Processing:** Hỗ trợ xử lý song song nhiều nguồn video (Multithreading) để tối ưu hiệu năng.
- **Reporting:** Tự động xuất báo cáo JSON summary và video đã được gán nhãn (annotated).

---

## 2. Tính năng chính

### ✅ Phát hiện đối tượng (Object Detection)
- Phát hiện phương tiện sử dụng **YOLO11**.
- Phân loại 4 loại phương tiện chính: `car`, `motorcycle`, `bus`, `truck`.
- Đếm số lượng phương tiện theo từng loại trong khung hình.

### ✅ Theo dõi đối tượng (Multi-Object Tracking)
- Tích hợp **ByteTrack** để theo dõi đa đối tượng.
- Gán và duy trì ID duy nhất (Unique ID) cho mỗi phương tiện trong suốt quá trình di chuyển.
- Lưu trữ và hiển thị lịch sử di chuyển (trajectory) của từng xe.

### ✅ Phân tích hành vi & Giao thông
- **Phát hiện đi ngược chiều (Wrong-way detection):** Cảnh báo ngay lập tức khi phương tiện di chuyển sai hướng quy định.
- **Phân chia làn đường:** Hỗ trợ định nghĩa ROI (Region of Interest) và Lane Divider.
- **Ước tính tốc độ:** Chuyển đổi tọa độ sang bird's-eye view để tính toán vận tốc thực tế (km/h).

### ✅ Tool cấu hình Zone trực quan
- **Setup Tool**: Giao diện GUI (OpenCV) tương tác để vẽ zone, chọn điểm perspective, kẻ vạch phân làn.
- Hỗ trợ lưu/load cấu hình (JSON) cho từng camera/video khác nhau.

---

## 3. Demo

### Single Stream
![Single Stream](demo/single_stream.png)

### Multi Stream (2x2 Grid)
![Multi Stream](demo/multi_stream.png)

---

## 4. Cài đặt & Hướng dẫn sử dụng

### Yêu cầu hệ thống (Prerequisites)
- Python 3.8+
- GPU (Recommended CUDA) hoặc CPU
- Các thư viện: `ultralytics`, `opencv-python`, `numpy`...

### Cài đặt (Installation)

```bash
pip install -r requirements.txt
```

### Cấu trúc project

```
vehicle_tracking/
├── main.py                 # Entry point chính
├── tracker.py              # Core logic: Vehicle tracking & detection
├── stream_processor.py     # Xử lý luồng video (Single/Threaded)
├── multi_stream.py         # Quản lý Multi-stream
├── setup_zones.py          # Tool cấu hình Zone/ROI
├── extract_frames.py       # Tool trích xuất frames (dataset preparation)
├── utils/
│   └── zone_loader.py      # Module load config
├── configs/                # File cấu hình JSON cho từng video
├── outputs/                # Thư mục chứa kết quả (Video + Report)
└── yolo11n.pt              # Model YOLO weights
```

### Hướng dẫn sử dụng

#### Bước 1: Cấu hình Zone (Lần đầu hoặc khi đổi góc camera)
Chạy tool setup để định nghĩa vùng quan tâm và góc quay:
```bash
python setup_zones.py --video Option1/Road_1.mp4 --mode all
```
*Controls: Click để chọn điểm, `C` để clear, `S` để lưu.*

#### Bước 2: Chạy phân tích (Single Video)
```bash
python main.py run --video Road_1.mp4
```

#### Bước 3: Chạy chế độ Multi-stream (Giám sát nhiều camera)
```bash
python main.py run --multi
```

#### Trích xuất dữ liệu để finetune YOLO model (Optional)
```bash
python extract_frames.py --random 50
```

---

## 5. Output

Hệ thống sẽ xuất dữ liệu vào thư mục `outputs/`:
1. **Video Result (.mp4):** Visualized với bounding boxes, thông tin xe, cảnh báo, tốc độ.
2. **JSON Summary:** Thống kê tổng hợp (tổng lưu lượng, số lượng từng loại xe, số vi phạm...).

```json
{
  "stream": "Road_1",
  "frames": 1500,
  "fps": 45.2,
  "total": 156,
  "counts": {"car": 120, "bus": 15, "truck": 18, "motorcycle": 3},
  "wrong_way": 2
}
```

---
