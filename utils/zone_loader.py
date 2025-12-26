"""
Zone Configuration Loader
=========================
Helper module để load và xử lý zone configs trong main tracking app.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ZoneConfig:
    """Class để load và xử lý zone configuration cho một video."""
    
    def __init__(self, config_path: str, target_resolution: Tuple[int, int] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._perspective_matrix = None
        self._inverse_matrix = None
        
        # Scale ratio nếu target resolution khác config resolution
        self._scale_x = 1.0
        self._scale_y = 1.0
        if target_resolution:
            orig_res = self.resolution
            self._scale_x = target_resolution[0] / orig_res[0]
            self._scale_y = target_resolution[1] / orig_res[1]
    
    def _load_config(self) -> dict:
        """Load config từ JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    @property
    def video_name(self) -> str:
        return self.config.get('video', '')
    
    @property
    def resolution(self) -> Tuple[int, int]:
        res = self.config.get('resolution', [1280, 720])
        return tuple(res)
    
    def _scale_points(self, pts: list) -> np.ndarray:
        """Scale points theo target resolution."""
        scaled = [[p[0] * self._scale_x, p[1] * self._scale_y] for p in pts]
        return np.float32(scaled)
    
    @property
    def perspective_points(self) -> np.ndarray:
        """Get perspective points as numpy array (scaled)."""
        pts = self.config.get('perspective_points', [])
        if len(pts) != 4:
            return None
        return self._scale_points(pts)
    
    @property
    def roi_points(self) -> np.ndarray:
        """Get ROI points as numpy array (scaled)."""
        pts = self.config.get('roi_points', [])
        if len(pts) != 4:
            return None
        return self._scale_points(pts)
    
    @property
    def lane_line(self) -> List[List[int]]:
        """Get lane divider line (scaled)."""
        pts = self.config.get('lane_line', [])
        if not pts:
            return []
        return [[int(p[0] * self._scale_x), int(p[1] * self._scale_y)] for p in pts]
    
    @property
    def lanes(self) -> dict:
        """Get lane configurations."""
        return self.config.get('lanes', {
            'left': {'expected_direction': 'down'},
            'right': {'expected_direction': 'up'}
        })
    
    def get_perspective_matrix(self, dst_size: Tuple[int, int] = (200, 400)) -> np.ndarray:
        """
        Get perspective transform matrix.
        
        Args:
            dst_size: (width, height) of destination rectangle
            
        Returns:
            3x3 perspective transform matrix
        """
        if self._perspective_matrix is not None:
            return self._perspective_matrix
        
        src_pts = self.perspective_points
        if src_pts is None:
            raise ValueError("Perspective points not configured")
        
        # Destination rectangle (bird's eye view)
        w, h = dst_size
        dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        self._perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return self._perspective_matrix
    
    def get_inverse_matrix(self, dst_size: Tuple[int, int] = (200, 400)) -> np.ndarray:
        """Get inverse perspective transform matrix."""
        if self._inverse_matrix is not None:
            return self._inverse_matrix
        
        src_pts = self.perspective_points
        if src_pts is None:
            raise ValueError("Perspective points not configured")
        
        w, h = dst_size
        dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        self._inverse_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        return self._inverse_matrix
    
    def transform_point(self, point: Tuple[int, int], dst_size: Tuple[int, int] = (200, 400)) -> Tuple[float, float]:
        """
        Transform a point from camera view to bird's eye view.
        
        Args:
            point: (x, y) in camera coordinates
            dst_size: destination rectangle size
            
        Returns:
            (x, y) in bird's eye coordinates
        """
        matrix = self.get_perspective_matrix(dst_size)
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, matrix)
        return tuple(transformed[0][0])
    
    def transform_points(self, points: np.ndarray, dst_size: Tuple[int, int] = (200, 400)) -> np.ndarray:
        """
        Transform multiple points from camera view to bird's eye view.
        
        Args:
            points: Nx2 array of points
            dst_size: destination rectangle size
            
        Returns:
            Nx2 array of transformed points
        """
        matrix = self.get_perspective_matrix(dst_size)
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(pts, matrix)
        return transformed.reshape(-1, 2)
    
    def is_in_roi(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the ROI polygon."""
        roi = self.roi_points
        if roi is None:
            return True  # No ROI = accept all
        
        result = cv2.pointPolygonTest(roi.astype(np.int32), point, False)
        return result >= 0
    
    def get_lane(self, point: Tuple[int, int]) -> str:
        """
        Determine which lane a point belongs to.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            'left' or 'right'
        """
        lane_line = self.lane_line
        if len(lane_line) != 2:
            # No lane divider, use simple x-based split
            roi = self.roi_points
            if roi is not None:
                center_x = np.mean(roi[:, 0])
                return 'left' if point[0] < center_x else 'right'
            return 'left'  # Default
        
        # Use lane divider line
        # Line equation: (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        x1, y1 = lane_line[0]
        x2, y2 = lane_line[1]
        
        # Calculate which side of line the point is on
        d = (point[0] - x1) * (y2 - y1) - (point[1] - y1) * (x2 - x1)
        
        return 'left' if d < 0 else 'right'
    
    def get_expected_direction(self, lane: str) -> str:
        """Get expected movement direction for a lane."""
        return self.lanes.get(lane, {}).get('expected_direction', 'down')
    
    def is_wrong_way(self, lane: str, direction: str) -> bool:
        """
        Check if vehicle is going wrong way.
        
        Args:
            lane: 'left' or 'right'
            direction: 'up' or 'down' (detected movement direction)
            
        Returns:
            True if wrong way
        """
        expected = self.get_expected_direction(lane)
        return direction != expected


class ZoneManager:
    """Manager để load và quản lý multiple zone configs."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, ZoneConfig] = {}
    
    def load_config(self, video_name: str) -> ZoneConfig:
        """Load config for a specific video."""
        if video_name in self.configs:
            return self.configs[video_name]
        
        # Remove extension if present
        video_stem = Path(video_name).stem
        config_path = self.config_dir / f"{video_stem}.json"
        
        config = ZoneConfig(str(config_path))
        self.configs[video_name] = config
        return config
    
    def load_all(self) -> Dict[str, ZoneConfig]:
        """Load all configs in config directory."""
        if not self.config_dir.exists():
            return {}
        
        for config_file in self.config_dir.glob("*.json"):
            video_name = config_file.stem
            if video_name not in self.configs:
                self.configs[video_name] = ZoneConfig(str(config_file))
        
        return self.configs
    
    def get_config(self, video_name: str) -> Optional[ZoneConfig]:
        """Get loaded config for a video."""
        video_stem = Path(video_name).stem
        return self.configs.get(video_stem)


# Import cv2 at module level for transform functions
try:
    import cv2
except ImportError:
    print("[Warning] OpenCV not found. Some functions may not work.")


if __name__ == "__main__":
    # Test loading
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        config = ZoneConfig(config_path)
        print(f"Loaded config for: {config.video_name}")
        print(f"Resolution: {config.resolution}")
        print(f"Perspective points: {config.perspective_points}")
        print(f"ROI points: {config.roi_points}")
        print(f"Lane line: {config.lane_line}")
        print(f"Lanes: {config.lanes}")
    else:
        # Load all configs
        manager = ZoneManager()
        configs = manager.load_all()
        print(f"Loaded {len(configs)} configs:")
        for name, cfg in configs.items():
            print(f"  - {name}: {cfg.resolution}")
