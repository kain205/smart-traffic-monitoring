"""
Vehicle Tracker - Simplified
"""

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

from ultralytics import YOLO
from utils.zone_loader import ZoneConfig


VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

COLORS = {
    'car': (0, 255, 0), 'motorcycle': (255, 0, 0), 'bus': (0, 165, 255),
    'truck': (0, 255, 255), 'wrong_way': (0, 0, 255), 'roi': (255, 0, 255),
    'lane_line': (255, 255, 0),
}


@dataclass
class TrackedVehicle:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    lane: str = 'unknown'
    direction: str = 'unknown'
    is_wrong_way: bool = False
    speed_kmh: float = 0.0
    history: List[Tuple[int, int]] = field(default_factory=list)
    transformed_history: List[Tuple[float, float]] = field(default_factory=list)


class VehicleTracker:
    def __init__(self, model_path: str = "yolo11n.pt", zone_config: Optional[ZoneConfig] = None,
                 conf_threshold: float = 0.3, real_world_height_m: float = 30.0):
        self.model = YOLO(model_path)
        self.zone_config = zone_config
        self.conf_threshold = conf_threshold
        self.real_world_height_m = real_world_height_m  # Chiều dài thực của vùng perspective (mét)
        
        self.tracked: Dict[int, TrackedVehicle] = {}
        self.counts = defaultdict(int)
        self.wrong_way_count = 0
        self.counted_ids = set()
        
        self.fps = 0
        self.video_fps = 30  # Sẽ được cập nhật từ video
        self._frame_times = []
        
        # Perspective transform matrix
        self._perspective_matrix = None
        self._dst_size = (200, 400)  # Bird's eye view size
        self._setup_perspective()
    
    def _setup_perspective(self):
        """Setup perspective transform matrix từ config."""
        if self.zone_config and self.zone_config.perspective_points is not None:
            src_pts = self.zone_config.perspective_points
            w, h = self._dst_size
            dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            self._perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    def _transform_point(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        """Transform point sang bird's eye view."""
        if self._perspective_matrix is None:
            return pt
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(p, self._perspective_matrix)
        return tuple(transformed[0][0])
    
    def _calculate_speed(self, vehicle: TrackedVehicle) -> float:
        """Tính speed (km/h) dựa trên movement trong transformed space."""
        hist = vehicle.transformed_history
        if len(hist) < 2 or self.video_fps <= 0:
            return 0.0
        
        # Tính khoảng cách di chuyển trong N frames gần nhất
        n = min(5, len(hist))
        recent = hist[-n:]
        
        total_dist = 0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            total_dist += np.sqrt(dx*dx + dy*dy)
        
        # Convert pixel distance to meters
        # dst_size[1] pixels = real_world_height_m meters
        pixels_per_meter = self._dst_size[1] / self.real_world_height_m
        distance_m = total_dist / pixels_per_meter
        
        # Time = n-1 frames
        time_s = (n - 1) / self.video_fps
        if time_s <= 0:
            return 0.0
        
        speed_ms = distance_m / time_s
        speed_kmh = speed_ms * 3.6
        
        return min(speed_kmh, 150)  # Cap tại 150 km/h
    
    def _is_in_roi(self, pt: Tuple[int, int]) -> bool:
        if not self.zone_config or self.zone_config.roi_points is None:
            return True
        roi = self.zone_config.roi_points.astype(np.int32)
        return cv2.pointPolygonTest(roi, pt, False) >= 0
    
    def _get_lane(self, pt: Tuple[int, int]) -> str:
        """Xác định lane dựa trên vị trí so với lane_line."""
        if not self.zone_config or not self.zone_config.lane_line:
            return 'unknown'
        
        lane_line = self.zone_config.lane_line
        if len(lane_line) != 2:
            return 'unknown'
        
        # lane_line[0] = điểm dưới, lane_line[1] = điểm trên
        x1, y1 = lane_line[0]  # bottom
        x2, y2 = lane_line[1]  # top
        
        # Cross product để xác định bên nào của line
        # Nếu đường đi từ dưới lên (y giảm), thì bên trái có d > 0
        d = (pt[0] - x1) * (y2 - y1) - (pt[1] - y1) * (x2 - x1)
        
        return 'left' if d > 0 else 'right'
    
    def _get_direction(self, vehicle: TrackedVehicle) -> str:
        """Xác định hướng di chuyển dựa trên lịch sử centroid."""
        if len(vehicle.history) < 5:
            return 'unknown'
        
        # Lấy 5 điểm gần nhất
        recent = vehicle.history[-5:]
        
        # Tính tổng di chuyển theo Y
        total_dy = sum(recent[i][1] - recent[i-1][1] for i in range(1, len(recent)))
        
        if abs(total_dy) < 10:  # Threshold
            return 'unknown'
        
        # Trong camera view: Y tăng = đi xuống (down), Y giảm = đi lên (up)
        return 'down' if total_dy > 0 else 'up'
    
    def _check_wrong_way(self, vehicle: TrackedVehicle) -> bool:
        if not self.zone_config or vehicle.direction == 'unknown' or vehicle.lane == 'unknown':
            return False
        
        expected = self.zone_config.lanes.get(vehicle.lane, {}).get('expected_direction')
        if not expected:
            return False
        
        return vehicle.direction != expected
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TrackedVehicle]]:
        start = time.time()
        
        # YOLO tracking
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=self.conf_threshold, classes=list(VEHICLE_CLASSES.keys()), verbose=False
        )
        
        vehicles = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if not self._is_in_roi(centroid):
                    continue
                
                class_name = VEHICLE_CLASSES.get(cls_id, 'unknown')
                
                # Get or create vehicle
                if track_id in self.tracked:
                    v = self.tracked[track_id]
                    v.bbox = (x1, y1, x2, y2)
                    v.centroid = centroid
                else:
                    v = TrackedVehicle(track_id, class_name, (x1, y1, x2, y2), centroid)
                    self.tracked[track_id] = v
                
                # Update history
                v.history.append(centroid)
                if len(v.history) > 30:
                    v.history = v.history[-30:]
                
                # Update transformed history cho speed calculation
                transformed_pt = self._transform_point(centroid)
                v.transformed_history.append(transformed_pt)
                if len(v.transformed_history) > 30:
                    v.transformed_history = v.transformed_history[-30:]
                
                # Update lane, direction, wrong-way, speed
                v.lane = self._get_lane(centroid)
                v.direction = self._get_direction(v)
                v.is_wrong_way = self._check_wrong_way(v)
                v.speed_kmh = self._calculate_speed(v)
                
                # Count
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.counts[class_name] += 1
                    if v.is_wrong_way:
                        self.wrong_way_count += 1
                
                vehicles.append(v)
        
        # FPS
        self._frame_times.append(time.time() - start)
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]
        self.fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))
        
        # Draw
        annotated = self._draw(frame, vehicles)
        return annotated, vehicles
    
    def _draw(self, frame: np.ndarray, vehicles: List[TrackedVehicle]) -> np.ndarray:
        out = frame.copy()
        
        # ROI
        if self.zone_config and self.zone_config.roi_points is not None:
            cv2.polylines(out, [self.zone_config.roi_points.astype(np.int32)], True, COLORS['roi'], 2)
        
        # Lane line
        if self.zone_config and self.zone_config.lane_line and len(self.zone_config.lane_line) == 2:
            pt1, pt2 = self.zone_config.lane_line
            cv2.line(out, tuple(pt1), tuple(pt2), COLORS['lane_line'], 2)
        
        # Vehicles
        for v in vehicles:
            x1, y1, x2, y2 = v.bbox
            color = COLORS['wrong_way'] if v.is_wrong_way else COLORS.get(v.class_name, (255,255,255))
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            
            # Label
            arrow = {'up': '↑', 'down': '↓'}.get(v.direction, '?')
            label = f"ID:{v.track_id} {v.class_name} {arrow}"
            if v.speed_kmh > 5:
                label += f" {v.speed_kmh:.0f}km/h"
            if v.is_wrong_way:
                label += " WRONG!"
            
            cv2.putText(out, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Trajectory
            if len(v.history) > 1:
                pts = np.array(v.history, np.int32)
                cv2.polylines(out, [pts], False, color, 1)
        
        # Stats
        cv2.putText(out, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(out, f"Total: {sum(self.counts.values())}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        y = 90
        for cls, cnt in self.counts.items():
            cv2.putText(out, f"{cls}: {cnt}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(cls, (255,255,255)), 1)
            y += 20
        
        if self.wrong_way_count > 0:
            cv2.putText(out, f"WRONG WAY: {self.wrong_way_count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['wrong_way'], 2)
        
        return out
    
    def get_stats(self) -> dict:
        return {
            'fps': self.fps,
            'total': sum(self.counts.values()),
            'counts': dict(self.counts),
            'wrong_way': self.wrong_way_count,
        }


if __name__ == "__main__":
    config = ZoneConfig("configs/Road_2.json")
    tracker = VehicleTracker("yolo11n.pt", config)
    
    cap = cv2.VideoCapture("Option1/Road_2.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, tuple(config.resolution))
        annotated, _ = tracker.process_frame(frame)
        cv2.imshow("Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(tracker.get_stats())
