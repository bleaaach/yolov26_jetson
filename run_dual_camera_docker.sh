#!/bin/bash

# Clean up existing container
docker rm -f dual-camera-system 2>/dev/null

# Clean up possible dual_camera_system.py file
rm -f dual_camera_system.py 2>/dev/null

# Configuration
MODEL_DIR="/home/seeed/ultralytics_data"
# Performance optimization configuration
OPTIMIZED_IMGSZ=640  # Keep 640 to match TensorRT engine input
JPEG_QUALITY=60  # Reduce JPEG quality to reduce network transmission time
CAMERA_WIDTH=320  # Camera width
CAMERA_HEIGHT=240  # Camera height
CAMERA_FPS=30  # Camera frame rate

# Mixed quantization configuration - FP16 for detection, INT8 for pose estimation and segmentation
USE_MIXED_QUANTIZATION=true
FP16_MODELS="yolo26n"  # Detection models use FP16
INT8_MODELS="yolo26n-pose yolo26n-seg"  # Pose and segmentation models use INT8
BATCH_SIZE=1  # Batch size optimization
MAX_DETECTIONS=3  # Maximum detection limit

# Jetson performance optimization suggestions
echo "=== Jetson Performance Optimization Suggestions ==="
echo "Run the following commands for best performance:"
echo "  sudo nvpmodel -m 0  # Maximum performance mode"
echo "  sudo jetson_clocks  # Enable maximum clock frequency"
echo "  tegrastats  # Monitor system performance"
echo ""
echo "=== TensorRT FP16/INT8 Mixed Quantization Optimization Tips ==="
echo "Check if TensorRT engines use mixed quantization:"
echo "  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n.engine --fp16"
echo "  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n-pose.engine --int8"
echo ""
echo "If you need to re-export mixed quantization engines:"
echo "  # FP16 detection model (high precision detection)"
echo "  yolo export model=yolo26n.pt format=engine device=0 half=True"
echo "  # INT8 pose model (fast inference)"  
echo "  yolo export model=yolo26n-pose.pt format=engine device=0 int8=True"
echo "  # INT8 segmentation model (fast segmentation)"
echo "  yolo export model=yolo26n-seg.pt format=engine device=0 int8=True"
echo ""

echo "=== Dual USB Camera Image Processing System Startup Script ==="
echo "Based on YOLOv26 model and TensorRT acceleration"
echo ""
echo "System configuration:"
echo "- Camera 1: Object detection + Pose estimation"
echo "- Camera 2: Object detection + SAM model"
echo "- Web server: http://localhost:5000"
echo ""
echo "Model directory: $MODEL_DIR"
echo ""
echo "Press Ctrl+C to exit system"
echo ""

# Check model files
echo "=== Checking Model Files ==="
if [ ! -f "$MODEL_DIR/yolo26n.engine" ] || [ ! -f "$MODEL_DIR/yolo26n-pose.engine" ]; then
    echo "Error: Required model files do not exist"
    exit 1
fi

echo "✅ Model file check completed"
echo ""

# Create Python script file
cat > dual_camera_system.py << 'PYTHONEOF'
#!/usr/bin/env python3
"""
Dual USB Camera Image Processing System
Based on YOLOv26 model and TensorRT acceleration
"""

import cv2
import numpy as np
import os
import time
from threading import Thread
import queue

# GPU optimization functions
def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
    except ImportError:
        pass

def get_inference_stats():
    """Get inference statistics"""
    try:
        import torch
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            return {
                'memory_used_gb': memory_used,
                'memory_reserved_gb': memory_reserved
            }
    except ImportError:
        pass
    return None

# Configuration
MODEL_DIR = "/models"
DETECTION_MODEL = os.path.join(MODEL_DIR, "yolo26n.engine")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo26n-pose.engine")
SAM_MODEL = os.path.join(MODEL_DIR, "yolo26n-seg.engine")

# Performance optimization configuration
OPTIMIZED_IMGSZ = 640  # Keep 640 to match TensorRT engine input
JPEG_QUALITY = 60  # Reduce JPEG quality to reduce network transmission time
CAMERA_WIDTH = 320  # Camera width
CAMERA_HEIGHT = 240  # Camera height
CAMERA_FPS = 30  # Camera frame rate

# Mixed quantization configuration
USE_MIXED_QUANTIZATION = True
FP16_MODELS = ["yolo26n"]  # Detection models use FP16
INT8_MODELS = ["yolo26n-pose", "yolo26n-seg"]  # Pose and segmentation models use INT8
BATCH_SIZE = 1  # Batch size optimization
MAX_DETECTIONS = 3  # Maximum detection limit

# Pose keypoint threshold
POSE_KPT_THRESHOLD = 0.2

# COCO pose keypoint connections (17 points)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# Class list (for detection)
CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
           "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
           "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
           "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
           "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
           "toothbrush"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Custom post-processing functions
def custom_postprocess_pose(detection_results, pose_results, frame):
    """Custom pose post-processing function"""
    # Create a copy of result frame
    result_frame = frame.copy()
    
    # Process object detection results first
    for result in detection_results:
        # Get boxes
        boxes = result.boxes
        if boxes is not None:
            # Get all data at once, reduce CPU-GPU synchronization
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            
            # Iterate through each box
            for i in range(len(xyxy)):
                # Get boxes coordinates
                x1, y1, x2, y2 = xyxy[i]
                # Get confidence
                conf = confs[i]
                # Get class
                cls = clss[i]
                
                # Draw box
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw class and confidence
                label = f"{CLASSES[cls]}: {conf:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Then process pose estimation results
    for result in pose_results:
        # Get keypoints
        keypoints = result.keypoints
        if keypoints is not None:
            # Get all keypoint data at once, reduce CPU-GPU synchronization
            kpts = keypoints.data.cpu().numpy()
            
            # Iterate through each person
            for i in range(kpts.shape[0]):
                # Get keypoints coordinates
                pts = kpts[i, :, :2].astype(int)
                # Get confidence
                confs = kpts[i, :, 2]
                
                # Draw points
                for j, (x, y) in enumerate(pts):
                    if confs[j] > POSE_KPT_THRESHOLD:
                        cv2.circle(result_frame, (x, y), 3, (0, 255, 255), -1)
                
                # Draw connections
                for j, k in SKELETON:
                    if confs[j] > POSE_KPT_THRESHOLD and confs[k] > POSE_KPT_THRESHOLD:
                        cv2.line(result_frame, tuple(pts[j]), tuple(pts[k]), (0, 200, 0), 2)
    
    return result_frame

def get_optimal_model_config(model_name):
    """Get optimal model configuration, including quantization strategy"""
    config = {
        'device': 0,  # GPU device
        'imgsz': OPTIMIZED_IMGSZ,
        'batch': BATCH_SIZE,
        'conf': 0.25,  # Confidence threshold
        'iou': 0.45,   # IoU threshold
        'max_det': MAX_DETECTIONS,
        'half': False,  # Default not use FP16
        'int8': False,  # Default not use INT8
        'optimize': True,  # Enable optimization
        'stream': True,   # Enable stream mode
        'agnostic_nms': False,  # Class-agnostic NMS
        'classes': [0] if 'pose' in model_name or 'seg' in model_name else None  # Pose and segmentation only detect person
    }
    
    # Set quantization strategy based on model type
    if USE_MIXED_QUANTIZATION:
        if any(fp16_model in model_name for fp16_model in FP16_MODELS):
            config['half'] = True  # FP16 quantization
            config['int8'] = False
        elif any(int8_model in model_name for int8_model in INT8_MODELS):
            config['half'] = False
            config['int8'] = True  # INT8 quantization
    
    return config

class PerformanceMonitor:
    """Performance monitor"""
    def __init__(self, name):
        self.name = name
        self.frame_times = []
        self.inference_times = []
        self.fps_history = []
        self.target_fps = CAMERA_FPS
        self.window_size = 30
    
    def add_frame_time(self, frame_time):
        """Add frame processing time"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def add_inference_time(self, inference_time):
        """Add inference time"""
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
    
    def add_fps(self, fps):
        """Add FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
    
    def get_stats(self):
        """Get performance statistics"""
        if not self.frame_times:
            return None
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return {
            'avg_frame_time_ms': avg_frame_time,
            'avg_inference_time_ms': avg_inference_time,
            'avg_fps': avg_fps,
            'fps_target': self.target_fps,
            'performance_ratio': avg_fps / self.target_fps if self.target_fps > 0 else 0
        }
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if stats:
            print(f"[{self.name}] Performance Stats - FPS: {stats['avg_fps']:.1f}/{stats['fps_target']}, "
                  f"Frame time: {stats['avg_frame_time_ms']:.1f}ms, "
                  f"Inference time: {stats['avg_inference_time_ms']:.1f}ms, "
                  f"Performance ratio: {stats['performance_ratio']*100:.1f}%")

def custom_postprocess_sam(detection_results, sam_results, frame):
    """Custom SAM post-processing function"""
    # Create a copy of result frame
    result_frame = frame.copy()
    
    # Process object detection results first
    for result in detection_results:
        # Get boxes
        boxes = result.boxes
        if boxes is not None:
            # Get all data at once, reduce CPU-GPU synchronization
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            
            # Iterate through each box
            for i in range(len(xyxy)):
                # Get boxes coordinates
                x1, y1, x2, y2 = xyxy[i]
                # Get confidence
                conf = confs[i]
                # Get class
                cls = clss[i]
                
                # Draw box
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw class and confidence
                label = f"{CLASSES[cls]}: {conf:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Then process SAM model results
    for result in sam_results:
        # Get segmentation
        masks = result.masks
        if masks is not None:
            # Get all segmentation data at once, reduce CPU-GPU synchronization
            mask_data = masks.data.cpu().numpy()
            
            # Iterate through each segmentation
            for i in range(mask_data.shape[0]):
                # Get mask
                mask = mask_data[i]
                # Apply mask
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                color = COLORS[i % len(COLORS)]
                result_frame[mask > 0.5] = result_frame[mask > 0.5] * 0.7 + color * 0.3
    
    return result_frame

class CameraProcessor:
    """Camera processor base class"""
    def __init__(self, camera_id, name):
        """Initialize camera processor"""
        self.camera_id = camera_id
        self.name = name
        self.cap = None
        self.running = False
        self.fps = 0
        self.inference_time = 0
        self.last_time = time.time()
        self.frame_count = 0
        self.model_loaded = False
        self.detection_model = None
        self.secondary_model = None
        self.latest_result = None  # Store latest processing result
        self.infer_busy = False  # Inference busy flag, used for frame dropping strategy
        self.print_counter = 0  # Print counter, reduce output frequency
        self.performance_monitor = PerformanceMonitor(name)  # Performance monitor
    
    def start_capture(self):
        """Start camera capture"""
        print("=== Attempting to open camera %d (%s) ===" % (self.camera_id, self.name))
        
        # Try using GStreamer pipeline (optimized for Jetson)
        try:
            # Create different GStreamer pipelines for different camera IDs
            device_path = f"/dev/video{self.camera_id}"
            gst_str = (
                f"v4l2src device={device_path} ! "
                f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, framerate={CAMERA_FPS}/1 ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! appsink drop=true sync=false"
            )
            
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                print("✅ Camera %d (%s) opened successfully (GStreamer)" % (self.camera_id, self.name))
                print("Camera parameters: 320x240 @ 30fps")
                
                # Test read a few frames
                for i in range(5):
                    time.sleep(0.3)
                    ret, frame = self.cap.read()
                    if ret:
                        print("✅ Camera %d test read successful" % self.camera_id)
                        return True
                    else:
                        print("⚠️ Camera %d test read failed" % self.camera_id)
                
                self.cap.release()
        except Exception as e:
            print("⚠️ GStreamer open failed: %s" % e)
            if self.cap:
                self.cap.release()
        
        # Fallback to standard V4L2
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                print("✅ Camera %d (%s) opened successfully (V4L2)" % (self.camera_id, self.name))
                
                # Optimize camera parameters
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure to reduce latency
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance to reduce latency
                
                # Verify camera parameters
                width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                print("Camera parameters: %dx%d @ %dfps" % (int(width), int(height), fps))
                
                # Test read a few frames
                for i in range(5):
                    time.sleep(0.3)
                    ret, frame = self.cap.read()
                    if ret:
                        print("✅ Camera %d test read successful" % self.camera_id)
                        return True
                    else:
                        print("⚠️ Camera %d test read failed" % self.camera_id)
                
                self.cap.release()
        except Exception as e:
            print("⚠️ V4L2 open failed: %s" % (e))
            if self.cap:
                self.cap.release()
        
        # All attempts failed
        raise RuntimeError("Cannot open camera %d" % self.camera_id)
    
    def capture_and_process_frames(self):
        """Thread for capturing and processing camera frames"""
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and self.model_loaded:
                        # Drop frame directly when inference is busy, avoid latency explosion
                        if self.infer_busy:
                            continue
                        # Process frame directly, do not use queue
                        self.process_frame(frame)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print("Camera %d processing error: %s" % (self.camera_id, e))
                time.sleep(1)
    
    def process_frame(self, frame):
        """Process single frame"""
        # Set inference busy flag
        self.infer_busy = True
        
        # Detailed time measurement
        total_start = time.time()
        
        # Preprocessing time (included in model call)
        model_start = time.time()
        # Use TensorRT optimized inference
        detection_results = self.detection_model(frame, imgsz=OPTIMIZED_IMGSZ, device=0)
        pose_results = self.secondary_model(frame, task='pose', imgsz=OPTIMIZED_IMGSZ, device=0)
        model_time = (time.time() - model_start) * 1000
        
        # Post-processing time
        postprocess_start = time.time()
        
        # Measure post-processing method execution time separately
        plot_start = time.time()
        # Use custom post-processing function instead of plot() method, pass object detection results
        pose_frame = custom_postprocess_pose(detection_results, pose_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(pose_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # Total time
        total_time = (time.time() - total_start) * 1000
        
        # Update FPS
        self.update_fps()
        
        # Reduce print frequency, print every 30 frames
        self.print_counter += 1
        if self.print_counter >= 30:
            print(f"Camera1 - Total time: {total_time:.1f}ms, Model time: {model_time:.1f}ms, Post-processing time: {postprocess_time:.1f}ms, plot time: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # Store latest results
        self.latest_result = pose_frame
        
        # Clear inference busy flag
        self.infer_busy = False
    
    def load_detection_model(self):
        """Load object detection model"""
        try:
            from ultralytics import YOLO
            # Get optimal configuration
            config = get_optimal_model_config(DETECTION_MODEL)
            self.detection_model = YOLO(DETECTION_MODEL)
            # Apply optimization configuration - no need to call fuse() for TensorRT engine files
            if DETECTION_MODEL.endswith('.pt'):
                self.detection_model.model.fuse()  # Fuse layers to optimize inference
            print(f"✅ Detection model loaded - FP16: {config['half']}, INT8: {config['int8']}")
            return True
        except Exception as e:
            print("❌ Object detection model loading failed: %s" % e)
            return False
    
    def load_secondary_model(self):
        """Load secondary model (implemented by subclass)"""
        pass
    
    def load_models(self):
        """Load models"""
        try:
            # For Camera1, Pose model already includes detection functionality, no need to load detection model separately
            if self.name == "Camera1":
                if not self.load_secondary_model():
                    return
            else:
                if not self.load_detection_model() or not self.load_secondary_model():
                    return
            
            self.model_loaded = True
            print("✅ %s model loaded" % self.name)
        except Exception as e:
            print("❌ %s model loading failed: %s" % (self.name, e))
            self.model_loaded = False
    
    def start(self):
        """Start processor"""
        self.running = True
        
        # Open camera
        self.start_capture()
        
        # Load models
        self.load_models()
        
        # Start capture and processing threads
        Thread(target=self.capture_and_process_frames, daemon=True).start()
        
        print("✅ Camera %d (%s) processing thread started" % (self.camera_id, self.name))
    
    def stop(self):
        """Stop processor"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        print("✅ Camera %d (%s) stopped" % (self.camera_id, self.name))
    
    def get_result(self):
        """Get processing result"""
        return self.latest_result
    
    def get_result_nonblock(self):
        """Non-blocking get processing result"""
        return self.latest_result
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.last_time = current_time
            self.frame_count = 0

class Camera1Processor(CameraProcessor):
    """Camera1 processor: Object detection + Pose estimation"""
    def __init__(self):
        super().__init__(0, "Camera1")
    
    def load_secondary_model(self):
        """Load pose estimation model"""
        try:
            from ultralytics import YOLO
            # Get optimal configuration
            config = get_optimal_model_config(POSE_MODEL)
            self.secondary_model = YOLO(POSE_MODEL)
            # Apply optimization configuration - no need to call fuse() for TensorRT engine files
            if POSE_MODEL.endswith('.pt'):
                self.secondary_model.model.fuse()  # Fuse layers to optimize inference
            # No need to load detection model separately, Pose model already includes detection functionality
            self.detection_model = None
            print(f"✅ Pose model loaded - FP16: {config['half']}, INT8: {config['int8']}")
            return True
        except Exception as e:
            print("❌ Pose estimation model loading failed: %s" % e)
            return False
    
    def process_frame(self, frame):
        """Process single frame"""
        # Set inference busy flag
        self.infer_busy = True
        
        # Detailed time measurement
        total_start = time.time()
        
        # Preprocessing time (included in model call)
        model_start = time.time()
        # Only run Pose model inference once, Pose model already includes detection functionality
        # Use optimized configuration for inference
        pose_config = get_optimal_model_config(POSE_MODEL)
        pose_results = list(self.secondary_model(frame, task='pose', **pose_config))
        model_time = (time.time() - model_start) * 1000
        
        # Post-processing time
        postprocess_start = time.time()
        
        # Measure post-processing method execution time separately
        plot_start = time.time()
        # Extract detection boxes and keypoints from pose_results for drawing
        pose_frame = custom_postprocess_pose(pose_results, pose_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(pose_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # Total time
        total_time = (time.time() - total_start) * 1000
        
        # Update FPS
        self.update_fps()
        
        # Record performance data
        self.performance_monitor.add_frame_time(total_time)
        self.performance_monitor.add_inference_time(model_time)
        self.performance_monitor.add_fps(self.fps)
        
        # Periodically optimize GPU memory and print performance statistics
        if self.frame_count % 100 == 0:
            optimize_gpu_memory()
            stats = get_inference_stats()
            if stats:
                print(f"GPU memory usage: {stats['memory_used_gb']:.2f}GB / {stats['memory_reserved_gb']:.2f}GB")
            self.performance_monitor.print_stats()
        
        # Reduce print frequency, print every 30 frames
        self.print_counter += 1
        if self.print_counter >= 30:
            print(f"Camera1 - Total time: {total_time:.1f}ms, Pose model: {pose_results[0].speed['inference']:.1f}ms, Post-processing: {postprocess_time:.1f}ms, plot: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # Store latest results
        self.latest_result = pose_frame
        
        # Clear inference busy flag
        self.infer_busy = False

class Camera2Processor(CameraProcessor):
    """Camera2 processor: Object detection + SAM model"""
    def __init__(self):
        super().__init__(1, "Camera2")
        self.frame_id = 0  # Frame counter, used for SAM low-frequency trigger
        self.last_sam = None  # Cache previous SAM results
    
    def load_secondary_model(self):
        """Load SAM model"""
        try:
            from ultralytics import YOLO
            if os.path.exists(SAM_MODEL):
                # Get optimal configuration
                config = get_optimal_model_config(SAM_MODEL)
                self.secondary_model = YOLO(SAM_MODEL)
                # Apply optimization configuration - no need to call fuse() for TensorRT engine files
                if SAM_MODEL.endswith('.pt'):
                    self.secondary_model.model.fuse()  # Fuse layers to optimize inference
                print(f"✅ SAM model loaded - FP16: {config['half']}, INT8: {config['int8']}")
            else:
                print("⚠️ SAM model does not exist, will use object detection segmentation results")
            return True
        except Exception as e:
            print("❌ SAM model loading failed: %s" % e)
            return True
    
    def process_frame(self, frame):
        """Process single frame"""
        # Set inference busy flag
        self.infer_busy = True
        
        # Detailed time measurement
        total_start = time.time()
        
        # Model time (includes preprocessing and inference)
        model_start = time.time()
        
        # Get optimal configuration
        detection_config = get_optimal_model_config(DETECTION_MODEL)
        
        # Run object detection model first, run every frame
        detection_results = list(self.detection_model(frame, **detection_config))
        
        # SAM model changed to low-frequency trigger: run every 5 frames
        self.frame_id += 1
        SAM_INTERVAL = 5  # SAM model runs every N frames
        
        if self.secondary_model and self.frame_id % SAM_INTERVAL == 0:
            try:
                sam_config = get_optimal_model_config(SAM_MODEL)
                sam_results = list(self.secondary_model(frame, **sam_config))
                self.last_sam = sam_results  # Cache SAM results
            except Exception as e:
                print("SAM inference failed: %s" % e)
        
        # Use cached SAM results
        sam_results = self.last_sam if self.last_sam is not None else detection_results
        
        model_time = (time.time() - model_start) * 1000
        
        # Post-processing time
        postprocess_start = time.time()
        
        # Measure post-processing method execution time separately
        plot_start = time.time()
        # Use custom post-processing function instead of plot() method, pass object detection results
        result_frame = custom_postprocess_sam(detection_results, sam_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(result_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # Total time
        total_time = (time.time() - total_start) * 1000
        
        # Update FPS
        self.update_fps()
        
        # Periodically optimize GPU memory
        if self.frame_count % 100 == 0:
            optimize_gpu_memory()
            stats = get_inference_stats()
            if stats:
                print(f"GPU memory usage: {stats['memory_used_gb']:.2f}GB / {stats['memory_reserved_gb']:.2f}GB")
        
        # Reduce print frequency, print every 30 frames
        self.print_counter += 1
        if self.print_counter >= 30:
            sam_status = "New" if self.frame_id % SAM_INTERVAL == 0 else "Cached"
            print(f"Camera2 - Total time: {total_time:.1f}ms, Detection model: {detection_results[0].speed['inference']:.1f}ms, SAM: {sam_status}, Post-processing: {postprocess_time:.1f}ms, plot: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # Store latest results
        self.latest_result = result_frame
        
        # Clear inference busy flag
        self.infer_busy = False

class DualCameraSystem:
    """Dual camera system"""
    def __init__(self):
        """Initialize dual camera system"""
        self.camera1 = None
        self.camera2 = None
        self.camera1_available = False
        self.camera2_available = False
        self.running = False
    
    def find_available_cameras(self):
        """Find available cameras"""
        print("=== Finding available cameras ===")
        
        available_cameras = []
        for idx in [0, 1, 2, 3, 4]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    time.sleep(0.3)
                    ret, frame = cap.read()
                    if ret:
                        print("✅ Camera index %d available" % idx)
                        available_cameras.append(idx)
                    cap.release()
                else:
                    cap.release()
            except Exception:
                pass
        
        print("Found %d available cameras: %s" % (len(available_cameras), available_cameras))
        return available_cameras
    
    def start_web_server(self):
        """Start web server"""
        try:
            from flask import Flask, Response
            import threading
            import time
            
            app = Flask(__name__)
            
            def generate_frames(camera_id):
                """Generate video stream frames"""
                while True:
                    if camera_id == 1 and self.camera1_available:
                        frame = self.camera1.get_result_nonblock()
                        if frame is not None:
                            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                            if ret:
                                yield (b'--frame\r\n' 
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    elif camera_id == 2 and self.camera2_available:
                        frame = self.camera2.get_result_nonblock()
                        if frame is not None:
                            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                            if ret:
                                yield (b'--frame\r\n' 
                                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            @app.route('/')
            def index():
                """Home page"""
                return '''
                <html>
                <head>
                    <title>Dual Camera Image Processing System</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #333; }
                        .camera-container { margin: 20px 0; }
                        h2 { color: #666; }
                        img { border: 1px solid #ddd; padding: 10px; }
                    </style>
                </head>
                <body>
                    <h1>Dual Camera Image Processing System</h1>
                    <p>Based on YOLOv26 model and TensorRT acceleration</p>
                    
                    <div class="camera-container">
                        <h2>Camera 1: Object detection + Pose estimation</h2>
                        <img src="/video_feed/1" width="640">
                    </div>
                    
                    <div class="camera-container">
                        <h2>Camera 2: Object detection + SAM model</h2>
                        <img src="/video_feed/2" width="640">
                    </div>
                </body>
                </html>
                '''
            
            @app.route('/video_feed/<int:camera_id>')
            def video_feed(camera_id):
                """Video stream"""
                return Response(generate_frames(camera_id),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # Start web server
            threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, 
                                                  debug=False, threaded=True), 
                          daemon=True).start()
            
            print("✅ Web server started successfully")
            print("Please visit in browser: http://localhost:5000")
            
        except Exception as e:
            print("❌ Web server startup failed: %s" % e)
    
    def start(self):
        """Start dual camera system"""
        print("=== Dual USB Camera Image Processing System Startup ===")
        print("Based on YOLOv26 model and TensorRT acceleration")
        print("Camera 1: Object detection + Pose estimation")
        print("Camera 2: Object detection + SAM model")
        print("")
        
        # Find available cameras
        available_cameras = self.find_available_cameras()
        
        if len(available_cameras) == 0:
            print("❌ No available cameras, system cannot start")
            return
        
        # Start camera 1
        if len(available_cameras) >= 1:
            print("Starting camera 1...")
            try:
                self.camera1 = Camera1Processor()
                self.camera1.camera_id = available_cameras[0]
                self.camera1.start()
                print("✅ Camera 1 started successfully")
                self.camera1_available = True
            except Exception as e:
                print("❌ Camera 1 startup failed: %s" % e)
        
        # Start camera 2
        if len(available_cameras) >= 2:
            print("Starting camera 2...")
            try:
                self.camera2 = Camera2Processor()
                self.camera2.camera_id = available_cameras[1]
                self.camera2.start()
                print("✅ Camera 2 started successfully")
                self.camera2_available = True
            except Exception as e:
                print("❌ Camera 2 startup failed: %s" % e)
        else:
            print("⚠️ Not enough available cameras, only start camera 1")
        
        # Check if there are available cameras
        if not self.camera1_available and not self.camera2_available:
            print("❌ No available cameras, system cannot start")
            return
        
        # Start web server
        self.start_web_server()
        
        self.running = True
        print("")
        print("=== Dual camera system startup completed ===")
        print("Available cameras: %s%s" % ("Camera1 " if self.camera1_available else "", "Camera2" if self.camera2_available else ""))
        print("Press Ctrl+C to exit system")
        print("")
        
        # Main loop
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.running = False
                break
        
        # Stop system
        self.stop()
    
    def stop(self):
        """Stop dual camera system"""
        print("=== Stopping dual camera system ===")
        self.running = False
        
        if self.camera1_available:
            self.camera1.stop()
        if self.camera2_available:
            self.camera2.stop()
        
        print("=== Dual camera system stopped ===")

def main():
    """Main function"""
    try:
        # Check model files
        if not os.path.exists(DETECTION_MODEL) or not os.path.exists(POSE_MODEL):
            raise FileNotFoundError("Required model files do not exist")
        
        # Start dual camera system
        system = DualCameraSystem()
        system.start()
        
    except Exception as e:
        print("Error: %s" % e)

if __name__ == "__main__":
    main()
PYTHONEOF

# Start Docker container
docker run --privileged -it --name dual-camera-system --runtime=nvidia --gpus all --ipc=host --network=host \
    --device=/dev/video0 \
    --device=/dev/video1 \
    --device=/dev/video2 \
    --device=/dev/video3 \
    -v "$MODEL_DIR:/models" \
    -v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py" \
    ultralytics/ultralytics:latest-jetson-jetpack6 \
    bash -c "pip install flask --quiet && python3 /app/dual_camera_system.py"

# Clean up temporary files
rm -f dual_camera_system.py
