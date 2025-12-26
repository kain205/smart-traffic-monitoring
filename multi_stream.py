"""
Multi-Stream Manager - Simplified
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List
from dataclasses import dataclass

from stream_processor import ThreadedProcessor


@dataclass  
class StreamConfig:
    video: str
    config: str
    id: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = Path(self.video).stem


class MultiStreamManager:
    def __init__(self, streams: List[StreamConfig], model: str = "yolo11n.pt",
                 output_dir: str = "outputs", resolution: tuple = (640, 360), grid_cols: int = 2):
        
        self.streams = streams
        self.model = model
        self.output_dir = output_dir
        self.resolution = resolution
        self.grid_cols = grid_cols
        self.grid_rows = (len(streams) + grid_cols - 1) // grid_cols
        
        self.processors = {}
        self.is_running = False
    
    def start(self):
        print(f"Starting {len(self.streams)} streams...")
        
        for cfg in self.streams:
            try:
                p = ThreadedProcessor(cfg.video, cfg.config, self.model,
                                      self.output_dir, self.resolution, save_video=True, stream_id=cfg.id)
                self.processors[cfg.id] = p
                p.start()
                print(f"  Started: {cfg.id}")
            except Exception as e:
                print(f"  Failed {cfg.id}: {e}")
        
        self.is_running = True
        self._display_loop()
    
    def _display_loop(self):
        grid_w = self.resolution[0] * self.grid_cols
        grid_h = self.resolution[1] * self.grid_rows
        
        cv2.namedWindow("Multi-Stream", cv2.WINDOW_NORMAL)
        
        while self.is_running:
            grid = np.zeros((grid_h, grid_w, 3), np.uint8)
            
            for i, cfg in enumerate(self.streams):
                row, col = i // self.grid_cols, i % self.grid_cols
                x, y = col * self.resolution[0], row * self.resolution[1]
                
                p = self.processors.get(cfg.id)
                if p and p.latest_frame is not None:
                    # Resize frame cho khớp cell size
                    resized = cv2.resize(p.latest_frame, self.resolution)
                    grid[y:y+self.resolution[1], x:x+self.resolution[0]] = resized
                else:
                    cv2.putText(grid, f"Waiting: {cfg.id}", (x+10, y+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("Multi-Stream", grid)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
            
            # Check if all done
            if all(not p.is_running for p in self.processors.values()):
                break
        
        self.stop()
    
    def stop(self):
        self.is_running = False
        for p in self.processors.values():
            p.is_running = False
        cv2.destroyAllWindows()
        
        print("\n=== SUMMARY ===")
        for sid, p in self.processors.items():
            print(f"[{sid}] {p.tracker.get_stats()}")


if __name__ == "__main__":
    from pathlib import Path
    
    streams = []
    for i in range(1, 6):  # Road_1 đến Road_5
        v = Path(f"Option1/Road_{i}.mp4")
        c = Path(f"configs/Road_{i}.json")
        if v.exists() and c.exists():
            streams.append(StreamConfig(str(v), str(c)))
    
    if streams:
        mgr = MultiStreamManager(streams)
        mgr.start()
    else:
        print("No configured streams found. Run setup_zones.py first.")
