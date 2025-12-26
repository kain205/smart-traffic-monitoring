"""
Stream Processor - Simplified
"""

import cv2
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from queue import Queue

from tracker import VehicleTracker
from utils.zone_loader import ZoneConfig


class StreamProcessor:
    def __init__(self, video_source: str, config_path: str, model_path: str = "yolo11n.pt",
                 output_dir: str = "outputs", resolution: tuple = (1280, 720),
                 save_video: bool = True, display: bool = True, stream_id: str = None):
        
        self.video_source = video_source
        self.resolution = resolution
        self.save_video = save_video
        self.display = display
        self.stream_id = stream_id or Path(video_source).stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load zone config với target resolution để scale ROI/perspective
        self.zone_config = ZoneConfig(config_path, target_resolution=resolution)
        self.tracker = VehicleTracker(model_path, self.zone_config)
        
        self.cap = None
        self.writer = None
        self.is_running = False
        self.frame_count = 0
    
    def process(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print(f"[{self.stream_id}] Cannot open: {self.video_source}")
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Cập nhật video FPS cho tracker để tính speed chính xác
        self.tracker.video_fps = fps
        
        if self.save_video:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"{self.stream_id}_{ts}.mp4"
            self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.resolution)
            print(f"[{self.stream_id}] Saving to: {out_path}")
        
        self.is_running = True
        print(f"[{self.stream_id}] Started - FPS: {fps}, Frames: {total}")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, self.resolution)
            annotated, _ = self.tracker.process_frame(frame)
            
            if self.writer:
                self.writer.write(annotated)
            
            if self.display:
                cv2.imshow(f"Stream: {self.stream_id}", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            self.frame_count += 1
            if self.frame_count % 200 == 0:
                pct = self.frame_count / total * 100 if total > 0 else 0
                stats = self.tracker.get_stats()
                print(f"[{self.stream_id}] {self.frame_count}/{total} ({pct:.0f}%) - "
                      f"FPS: {stats['fps']:.0f}, Total: {stats['total']}, Wrong: {stats['wrong_way']}")
        
        self.stop()
    
    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        if self.display:
            cv2.destroyAllWindows()
        
        # Save summary
        stats = self.tracker.get_stats()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {'stream': self.stream_id, 'frames': self.frame_count, **stats}
        with open(self.output_dir / f"{self.stream_id}_{ts}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[{self.stream_id}] Done - {stats}")


class ThreadedProcessor(StreamProcessor):
    """Stream processor chạy trong thread riêng."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, display=False, **kwargs)
        self.thread = None
        self.latest_frame = None
    
    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            return
        
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.tracker.video_fps = fps
        
        if self.save_video:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"{self.stream_id}_{ts}.mp4"
            self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, self.resolution)
        
        self.is_running = True
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, self.resolution)
            annotated, _ = self.tracker.process_frame(frame)
            self.latest_frame = annotated.copy()
            
            if self.writer:
                self.writer.write(annotated)
            
            self.frame_count += 1
        
        self.stop()


if __name__ == "__main__":
    p = StreamProcessor("Option1/Road_2.mp4", "configs/Road_2.json", display=True, save_video=True)
    p.process()
