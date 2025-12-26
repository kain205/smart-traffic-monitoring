"""
Enhanced Zone Setup Tool for Traffic CCTV Analysis
===================================================
CLI tool để configure perspective transform, ROI, và lane divider cho mỗi video.

Usage:
    python setup_zones.py --video Road_1.mp4 --mode perspective
    python setup_zones.py --video Road_1.mp4 --mode roi
    python setup_zones.py --video Road_1.mp4 --mode lane-divider
    python setup_zones.py --video Road_1.mp4 --mode all

Controls:
    - Left Click: Thêm điểm
    - C: Clear/Reset points hiện tại
    - Z: Undo điểm cuối cùng
    - S: Save và tiếp tục (nếu mode=all)
    - Space: Next frame
    - R: Random frame (nhảy đến frame ngẫu nhiên)
    - Q: Quit
"""

import cv2
import numpy as np
import argparse
import json
import os
from pathlib import Path


# ==================== CONFIGURATION ====================
DEFAULT_RESOLUTION = (1280, 720)
PADDING = 350  # Padding xung quanh frame để chọn điểm ngoài khung
CONFIG_DIR = Path("configs")
VIDEO_DIR = Path("Option1")

# Colors (BGR)
COLORS = {
    'perspective': (0, 255, 0),    # Green
    'roi': (255, 0, 0),            # Blue
    'lane_divider': (0, 255, 255), # Yellow
    'text': (255, 255, 255),       # White
    'point': (0, 0, 255),          # Red
}

MODE_CONFIGS = {
    'perspective': {
        'num_points': 4,
        'instruction': 'Chon 4 diem PERSPECTIVE theo chieu kim dong ho: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left',
        'key': 'perspective_points'
    },
    'roi': {
        'num_points': 4,
        'instruction': 'Chon 4 diem ROI (vung tracking) theo chieu kim dong ho: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left',
        'key': 'roi_points'
    },
    'lane-divider': {
        'num_points': 2,
        'instruction': 'Chon 2 diem LANE DIVIDER: diem tren -> diem duoi (chia lane trai/phai)',
        'key': 'lane_line'
    }
}


class ZoneSetupTool:
    def __init__(self, video_path: str, mode: str, resolution: tuple = DEFAULT_RESOLUTION):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.mode = mode
        self.resolution = resolution
        self.padding = PADDING
        self.canvas_size = (resolution[0] + 2 * PADDING, resolution[1] + 2 * PADDING)
        self.points = []
        self.current_mode_idx = 0
        self.modes_to_run = []
        self.config = self._load_existing_config()
        self.frame = None
        self.original_frame = None
        self.canvas = None  # Frame with padding
        self.cap = None
        
        # Setup modes to run
        if mode == 'all':
            self.modes_to_run = ['perspective', 'roi', 'lane-divider']
        else:
            self.modes_to_run = [mode]
    
    def _get_config_path(self) -> Path:
        """Get config file path for current video."""
        CONFIG_DIR.mkdir(exist_ok=True)
        return CONFIG_DIR / f"{self.video_name}.json"
    
    def _load_existing_config(self) -> dict:
        """Load existing config if exists."""
        config_path = self._get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'video': self.video_name,
            'resolution': list(self.resolution),
            'perspective_points': [],
            'roi_points': [],
            'lane_line': [],
            'lanes': {
                'left': {'expected_direction': 'down'},
                'right': {'expected_direction': 'up'}
            }
        }
    
    def _save_config(self):
        """Save config to JSON file."""
        config_path = self._get_config_path()
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"[OK] Config saved to: {config_path}")
    
    def _get_current_mode(self) -> str:
        """Get current mode being configured."""
        if self.current_mode_idx < len(self.modes_to_run):
            return self.modes_to_run[self.current_mode_idx]
        return None
    
    def _get_mode_config(self) -> dict:
        """Get config for current mode."""
        current_mode = self._get_current_mode()
        if current_mode:
            return MODE_CONFIGS[current_mode]
        return None
    
    def _draw_ui(self):
        """Draw UI overlay on frame with padding."""
        # Create canvas with padding (gray background)
        self.canvas = np.full((self.canvas_size[1], self.canvas_size[0], 3), 50, dtype=np.uint8)
        
        # Place original frame in center
        p = self.padding
        self.canvas[p:p+self.resolution[1], p:p+self.resolution[0]] = self.original_frame.copy()
        
        # Draw padding border
        cv2.rectangle(self.canvas, (p-1, p-1), 
                      (p+self.resolution[0], p+self.resolution[1]), (100, 100, 100), 2)
        
        current_mode = self._get_current_mode()
        mode_cfg = self._get_mode_config()
        
        if not mode_cfg:
            return
        
        # Draw instruction
        instruction = mode_cfg['instruction']
        cv2.putText(self.canvas, instruction, (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        # Draw progress
        progress = f"Mode: {current_mode.upper()} | Points: {len(self.points)}/{mode_cfg['num_points']}"
        cv2.putText(self.canvas, progress, (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        
        # Draw controls
        controls = "[C]lear | [Z]Undo | [S]ave | [Space]Next | [R]andom | [Q]uit"
        cv2.putText(self.canvas, controls, (10, self.canvas_size[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
        
        # Draw existing configs (faded)
        self._draw_existing_configs()
        
        # Draw current points (adjusted for padding)
        color = COLORS.get(current_mode.replace('-', '_'), COLORS['point'])
        for i, pt in enumerate(self.points):
            # Points are stored in original coords, need to add padding for display
            display_pt = (pt[0] + p, pt[1] + p)
            cv2.circle(self.canvas, display_pt, 8, COLORS['point'], -1)
            cv2.circle(self.canvas, display_pt, 10, color, 2)
            cv2.putText(self.canvas, str(i+1), (display_pt[0]+12, display_pt[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw lines connecting points
        if len(self.points) > 1:
            pts = np.array([[pt[0] + p, pt[1] + p] for pt in self.points], dtype=np.int32)
            if mode_cfg['num_points'] == 4 and len(self.points) == 4:
                # Draw closed polygon
                cv2.polylines(self.canvas, [pts], True, color, 2)
            elif mode_cfg['num_points'] == 2 and len(self.points) == 2:
                # Draw line
                cv2.line(self.canvas, tuple(pts[0]), tuple(pts[1]), color, 2)
            else:
                # Draw partial
                cv2.polylines(self.canvas, [pts], False, color, 2)
        
        # Show perspective preview if complete
        if current_mode == 'perspective' and len(self.points) == 4:
            self._draw_perspective_preview()
    
    def _draw_existing_configs(self):
        """Draw existing configured zones (faded)."""
        alpha = 0.3
        overlay = self.canvas.copy()
        p = self.padding
        
        # Draw perspective points
        if self.config.get('perspective_points') and self._get_current_mode() != 'perspective':
            pts = np.array([[pt[0] + p, pt[1] + p] for pt in self.config['perspective_points']], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, COLORS['perspective'], 2)
        
        # Draw ROI points
        if self.config.get('roi_points') and self._get_current_mode() != 'roi':
            pts = np.array([[pt[0] + p, pt[1] + p] for pt in self.config['roi_points']], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, COLORS['roi'], 2)
        
        # Draw lane divider
        if self.config.get('lane_line') and len(self.config['lane_line']) == 2 and self._get_current_mode() != 'lane-divider':
            pt1, pt2 = self.config['lane_line']
            cv2.line(overlay, (pt1[0] + p, pt1[1] + p), (pt2[0] + p, pt2[1] + p), COLORS['lane_divider'], 2)
        
        cv2.addWeighted(overlay, alpha, self.canvas, 1 - alpha, 0, self.canvas)
    
    def _draw_perspective_preview(self):
        """Draw small perspective transform preview."""
        src_pts = np.float32(self.points)
        # Destination: rectangle
        dst_pts = np.float32([[0, 0], [200, 0], [200, 300], [0, 300]])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(self.original_frame, matrix, (200, 300))
        
        # Place preview in corner of canvas
        preview_x = self.canvas_size[0] - 220
        preview_y = 80
        
        # Draw border
        cv2.rectangle(self.canvas, (preview_x-2, preview_y-2), 
                      (preview_x+202, preview_y+302), COLORS['perspective'], 2)
        cv2.putText(self.canvas, "Bird's Eye Preview", (preview_x, preview_y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['perspective'], 1)
        
        self.canvas[preview_y:preview_y+300, preview_x:preview_x+200] = warped
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            mode_cfg = self._get_mode_config()
            if mode_cfg and len(self.points) < mode_cfg['num_points']:
                # Convert canvas coords to original coords (subtract padding)
                orig_x = x - self.padding
                orig_y = y - self.padding
                self.points.append([orig_x, orig_y])
                print(f"Point {len(self.points)}: [{orig_x}, {orig_y}] (canvas: [{x}, {y}])")
                self._draw_ui()
                cv2.imshow("Zone Setup Tool", self.canvas)
    
    def _save_current_mode(self):
        """Save current mode's points to config."""
        mode_cfg = self._get_mode_config()
        if mode_cfg and len(self.points) == mode_cfg['num_points']:
            self.config[mode_cfg['key']] = self.points.copy()
            print(f"[OK] {self._get_current_mode()}: {self.points}")
            return True
        else:
            print(f"[!] Can du {mode_cfg['num_points']} diem de save!")
            return False
    
    def _next_mode(self):
        """Move to next mode."""
        self.current_mode_idx += 1
        self.points = []
        if self.current_mode_idx < len(self.modes_to_run):
            print(f"\n--- Chuyen sang mode: {self._get_current_mode().upper()} ---")
            return True
        return False
    
    def run(self):
        """Run the setup tool."""
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[X] Khong the mo video: {self.video_path}")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("[X] Khong the doc frame tu video")
            return
        
        self.original_frame = cv2.resize(frame, self.resolution)
        self.frame = self.original_frame.copy()
        
        # Setup window
        cv2.namedWindow("Zone Setup Tool")
        cv2.setMouseCallback("Zone Setup Tool", self._mouse_callback)
        
        print("\n" + "="*60)
        print(f"Video: {self.video_path}")
        print(f"Resolution: {self.resolution}")
        print(f"Mode: {self.mode}")
        print("="*60)
        print(f"\n--- Bat dau mode: {self._get_current_mode().upper()} ---")
        
        self._draw_ui()
        cv2.imshow("Zone Setup Tool", self.canvas)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                print("\nThoat...")
                break
            
            elif key == ord('c'):
                # Clear/Reset current points
                self.points = []
                print("[Clear] Points")
                self._draw_ui()
                cv2.imshow("Zone Setup Tool", self.canvas)
            
            elif key == ord('z'):
                # Undo last point
                if self.points:
                    removed = self.points.pop()
                    print(f"[Undo] Removed point: {removed}")
                    self._draw_ui()
                    cv2.imshow("Zone Setup Tool", self.canvas)
                else:
                    print("[Undo] No points to undo")
            
            elif key == ord('r'):
                # Random frame
                import random
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                random_frame = random.randint(0, total_frames - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.original_frame = cv2.resize(frame, self.resolution)
                    print(f"[Random] Frame {random_frame}/{total_frames}")
                    self._draw_ui()
                    cv2.imshow("Zone Setup Tool", self.canvas)
            
            elif key == ord('s'):
                # Save current mode
                if self._save_current_mode():
                    self._save_config()
                    if not self._next_mode():
                        print("\n[OK] Hoan thanh tat ca modes!")
                        break
                    self._draw_ui()
                    cv2.imshow("Zone Setup Tool", self.canvas)
            
            elif key == 32:  # Space - next frame
                ret, frame = self.cap.read()
                if ret:
                    self.original_frame = cv2.resize(frame, self.resolution)
                    self._draw_ui()
                    cv2.imshow("Zone Setup Tool", self.canvas)
                else:
                    # Loop back to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        self.original_frame = cv2.resize(frame, self.resolution)
                        self._draw_ui()
                        cv2.imshow("Zone Setup Tool", self.canvas)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final config
        print("\n" + "="*60)
        print("FINAL CONFIG:")
        print(json.dumps(self.config, indent=2))
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Zone Setup Tool for Traffic CCTV Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_zones.py --video Option1/Road_1.mp4 --mode perspective
    python setup_zones.py --video Option1/Road_2.mp4 --mode all
    python setup_zones.py --video Option1/Road_1.mp4 --mode roi --width 1920 --height 1080
        """
    )
    
    parser.add_argument('--video', '-v', type=str, required=True,
                        help='Path to video file (e.g., Road_1.mp4 or Option1/Road_1.mp4)')
    
    parser.add_argument('--mode', '-m', type=str, required=True,
                        choices=['perspective', 'roi', 'lane-divider', 'all'],
                        help='Setup mode: perspective, roi, lane-divider, or all')
    
    parser.add_argument('--width', '-W', type=int, default=1280,
                        help='Frame width (default: 1280)')
    
    parser.add_argument('--height', '-H', type=int, default=720,
                        help='Frame height (default: 720)')
    
    args = parser.parse_args()
    
    # Resolve video path
    video_path = args.video
    if not os.path.exists(video_path):
        # Try with VIDEO_DIR prefix
        video_path = str(VIDEO_DIR / args.video)
        if not os.path.exists(video_path):
            print(f"[X] Video khong ton tai: {args.video}")
            print(f"    Thu tim trong: {VIDEO_DIR}")
            return
    
    resolution = (args.width, args.height)
    
    tool = ZoneSetupTool(video_path, args.mode, resolution)
    tool.run()


if __name__ == "__main__":
    main()