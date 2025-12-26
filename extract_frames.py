"""
Frame Extractor cho Manual Labeling
====================================
Extract frames từ video để label bằng tool (LabelImg, CVAT, Roboflow, etc.)

Usage:
    python extract_frames.py                          # Extract từ tất cả video
    python extract_frames.py --video Road_1.mp4       # Extract từ 1 video
    python extract_frames.py --interval 50            # Mỗi 50 frames lấy 1
    python extract_frames.py --random 100             # Lấy random 100 frames mỗi video

Sau khi extract, dùng tool để label:
    - LabelImg: pip install labelImg && labelImg
    - CVAT: https://cvat.ai
    - Roboflow: https://roboflow.com (recommend, free tier có)
"""

import cv2
import random
import argparse
from pathlib import Path

# Config
VIDEOS = [
    "Option1/Road_1.mp4",
    "Option1/Road_2.mp4",
    "Option1/Road_3.mp4",
    "Option1/Road_4.mp4",
]

CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
OUTPUT_DIR = Path("dataset")


def extract_frames(video_path: str, interval: int = 30, random_count: int = None):
    """Extract frames từ video."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem
    
    # Tạo output dir
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Chọn frames để extract
    if random_count:
        frame_indices = sorted(random.sample(range(total_frames), min(random_count, total_frames)))
    else:
        frame_indices = list(range(0, total_frames, interval))
    
    print(f"\n[{video_name}] Extracting {len(frame_indices)} frames from {total_frames} total")
    
    extracted = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize về kích thước chuẩn
        frame = cv2.resize(frame, (1280, 720))
        
        # Lưu image
        img_name = f"{video_name}_{idx:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), frame)
        
        extracted += 1
        if extracted % 20 == 0:
            print(f"  Extracted: {extracted}/{len(frame_indices)}")
    
    cap.release()
    print(f"[{video_name}] Done: {extracted} frames")
    return extracted


def create_classes_txt():
    """Tạo classes.txt cho LabelImg."""
    classes_path = OUTPUT_DIR / "classes.txt"
    with open(classes_path, 'w') as f:
        for name in CLASS_NAMES:
            f.write(f"{name}\n")
    print(f"Created: {classes_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames for manual labeling")
    parser.add_argument('--video', '-v', help='Specific video to extract')
    parser.add_argument('--interval', '-i', type=int, default=100, help='Frame interval (default: 30)')
    parser.add_argument('--random', '-r', type=int, help='Random N frames per video')
    
    args = parser.parse_args()
    
    print("="*50)
    print("Frame Extractor for Manual Labeling")
    print("="*50)
    
    # Chọn videos
    videos = [args.video] if args.video else VIDEOS
    videos = [v for v in videos if Path(v).exists()]
    
    if not videos:
        print("No videos found!")
        return
    
    print(f"Videos: {videos}")
    print(f"Interval: {args.interval}, Random: {args.random}")
    
    # Extract
    total = 0
    for video in videos:
        count = extract_frames(video, args.interval, args.random)
        total += count or 0
    
    # Create classes.txt
    create_classes_txt()
    
    print("\n" + "="*50)
    print(f"Total frames extracted: {total}")
    print(f"Location: {OUTPUT_DIR.absolute()}/images/")
    print("\nNext steps:")
    print("  1. Label với LabelImg:")
    print(f"     pip install labelImg")
    print(f"     labelImg {OUTPUT_DIR}/images {OUTPUT_DIR}/classes.txt")
    print("  2. Hoặc upload lên Roboflow: https://roboflow.com")
    print("  3. Train sau khi label xong:")
    print("     yolo train model=yolo11n.pt data=dataset.yaml epochs=50")
    print("="*50)


if __name__ == "__main__":
    main()
