# Traffic Analysis Solution - Vehicle Tracking & Wrong-way Detection

---

## 1. Project Overview

A CCTV-based traffic analysis solution focusing on automated monitoring and traffic behavior analysis features:

- **Detection & Tracking:** Utilizes **YOLO11** combined with **ByteTrack** (state-of-the-art) to detect and track vehicles with high accuracy.
- **Wrong-way Detection:** Detects wrong-way vehicles based on Polygon Zones and expected movement direction.
- **Speed Estimation:** Estimates speed (km/h) using Perspective Transformation (homography) techniques.
- **Multi-stream Processing:** Supports parallel processing of multiple video sources (Multithreading) to optimize performance.
- **Reporting:** Automatically exports JSON summary reports and annotated videos.

---

## 2. Demo

### Single Stream
https://github.com/user-attachments/assets/7542b8bf-541b-456e-97eb-eed8597a997f



### Multi Stream 
https://github.com/user-attachments/assets/ea8ba138-6ef5-4b9f-bfe9-43005c54e719


---





## 3. Key Features

### ✅ Object Detection
- Detects vehicles using **YOLO11**.
- Classifies 4 main vehicle types: `car`, `motorcycle`, `bus`, `truck`.
- Counts the number of vehicles by type within the frame.

### ✅ Multi-Object Tracking
- Integrates **ByteTrack** for multi-object tracking.
- Assigns and maintains a Unique ID for each vehicle throughout its movement.
- Stores and displays the movement trajectory of each vehicle.

### ✅ Behavior & Traffic Analysis
- **Wrong-way Detection:** Immediately alerts when a vehicle moves in the wrong direction.
- **Lane Division:** Supports defining ROI (Region of Interest) and Lane Dividers.
- **Speed Estimation:** Converts coordinates to a bird's-eye view to calculate actual speed (km/h).

### ✅ Visual Zone Configuration Tool
- **Setup Tool:** Interactive GUI (OpenCV) to start drawing zones, selecting perspective points, and drawing lane dividers.
- Supports saving/loading configurations (JSON) for different cameras/videos.

---


## 4. Installation & Usage

### Prerequisites
- Python 3.8+
- GPU (Recommended CUDA) or CPU
- Libraries: `ultralytics`, `opencv-python`, `numpy`...

### Installation

```bash
pip install -r requirements.txt
```

### Project Structure

```
vehicle_tracking/
├── main.py                 # Main entry point
├── tracker.py              # Core logic: Vehicle tracking & detection
├── stream_processor.py     # Video stream processing (Single/Threaded)
├── multi_stream.py         # Multi-stream management
├── setup_zones.py          # Zone/ROI configuration tool
├── extract_frames.py       # Frame extraction tool (dataset preparation)
├── utils/
│   └── zone_loader.py      # Config loading module
├── configs/                # JSON configuration files for each video
├── outputs/                # Output directory (Video + Report)
└── yolo11n.pt              # YOLO model weights
```

### Usage Guide

#### Step 1: Zone Configuration (First time or when camera angle changes)
Run the setup tool to define the region of interest and perspective:
```bash
python setup_zones.py --video Option1/Road_1.mp4 --mode all
```
*Controls: Click to select points, `C` to clear, `S` to save.*

#### Step 2: Run Analysis (Single Video)
```bash
python main.py run --video Road_1.mp4
```

#### Step 3: Run Multi-stream Mode (Monitor multiple cameras)
```bash
python main.py run --multi
```

#### Extract Data for YOLO Finetuning (Optional)
```bash
python extract_frames.py --random 50
```

---

## 5. Output

The system will export data to the `outputs/` directory:
1. **Video Result (.mp4):** Visualized with bounding boxes, vehicle info, warnings, and speed.
2. **JSON Summary:** Aggregated statistics (total flow, count by vehicle type, violations...).

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
