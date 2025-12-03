"""
MCAP frame extraction
"""

import os
import numpy as np
from pathlib import Path
from mcap_ros2.reader import read_ros2_messages
from cv_bridge import CvBridge
import cv2

def extract_frames(mcap_path, output_dir):
    """Extract RGB and depth frames from MCAP file."""
    
    # Create output directories
    rgb_dir = Path(output_dir) / "rgb"
    depth_dir = Path(output_dir) / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    
    bridge = CvBridge()
    rgb_count = 0
    depth_count = 0
    
    # Read and save frames
    for msg_data in read_ros2_messages(str(mcap_path)):
        topic = msg_data.channel.topic
        msg = msg_data.ros_msg
        
        if topic == "/camera/camera/color/image_raw":
            # Save RGB frame
            img = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite(str(rgb_dir / f"{rgb_count:06d}.jpg"), img)
            rgb_count += 1
            
        if topic == "/camera/camera/depth/image_rect_raw":
            # raw depth 
            depth = bridge.imgmsg_to_cv2(msg, "passthrough")
            
            # clip depth
            depth_clipped = np.clip(depth, 50, 800)  
            visualize_depth = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # color map
            depth_colormap = cv2.applyColorMap(visualize_depth, cv2.COLORMAP_JET)
            
            # Save
            cv2.imwrite(str(depth_dir / f"{depth_count:06d}.jpg"), depth_colormap)
            
            depth_count += 1

if __name__ == "__main__":
    # Setup paths
    # Default to ./data and ./output relative to script location
    SCRIPT_DIR = Path(__file__).parent
    DATA_ROOT = Path(os.environ.get("DATA_ROOT", SCRIPT_DIR / "data"))
    OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", SCRIPT_DIR / "output"))
    
    # Process all mcap files
    for mcap_file in DATA_ROOT.rglob("*.mcap"):
        # mirror directory structure
        rel_path = mcap_file.parent.relative_to(DATA_ROOT)
        output_dir = OUTPUT_ROOT / rel_path
        
        print(f"Processing: {mcap_file.name}")
        extract_frames(mcap_file, output_dir)
    
    print("Done.")