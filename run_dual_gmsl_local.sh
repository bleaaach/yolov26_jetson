#!/bin/bash

# =============================================================================
# GMSL相机AGX Orin终极优化版 - 真正实现30fps
# 基于硬件能力重新设计优化策略
# =============================================================================

export OPENCV_IO_ENABLE_GSTREAMER=1
export CUDA_LAUNCH_BLOCKING=0

# 配置参数
MODEL_DIR="/home/seeed/yolov26_jetson/ultralytics_data"
OPTIMIZED_IMGSZ=320

# 目标：真正实现30fps
TARGET_FPS=30
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1536

# 处理分辨率（优化处理速度）
PROCESS_WIDTH=640
PROCESS_HEIGHT=480

# 显示分辨率
DISPLAY_WIDTH=640
DISPLAY_HEIGHT=480

echo "=== AGX Orin GMSL Camera Ultimate 30fps Edition ==="
echo "Target: ${TARGET_FPS}fps on AGX Orin"
echo "Capture: ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} @ ${TARGET_FPS}fps"
echo "Process: ${PROCESS_WIDTH}x${PROCESS_HEIGHT} (AGX Orin optimized)"
echo "Display: ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}"
echo ""

python3 << 'PYEOF'
import os
import cv2
import numpy as np
import time
import threading
import queue
import gi
import torch
import subprocess

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)
print("🚀 AGX Orin 30fps optimization system initialized")
cv2.setUseOptimized(True)

# AGX Orin性能优化
class AGXOrinOptimizer:
    """AGX Orin专用优化器"""
    
    @staticmethod
    def maximize_performance():
        """最大化AGX Orin性能"""
        try:
            if os.geteuid() == 0:
                subprocess.run(['nvpmodel', '-m', '0'], check=False)
                subprocess.run(['jetson_clocks'], check=False)
                for i in range(12):
                    try:
                        with open(f'/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor', 'w') as f:
                            f.write('performance')
                    except:
                        pass
            
            # 优化GPU
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # 设置CUDA设备
                torch.cuda.set_device(0)
                torch.cuda.empty_cache()
                
                # 获取GPU信息
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            print("✅ AGX Orin performance maximized")
            
        except Exception as e:
            print(f"Performance optimization error: {e}")
    
    @staticmethod
    def check_system_status():
        """检查系统状态"""
        try:
            # CPU频率
            cpu_freq = subprocess.run(['cat', '/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq'], 
                                    capture_output=True, text=True)
            if cpu_freq.returncode == 0:
                freq_mhz = int(cpu_freq.stdout.strip()) / 1000
                print(f"CPU Max Frequency: {freq_mhz}MHz")
            
            # GPU频率
            try:
                gpu_freq = subprocess.run(['cat', '/sys/devices/gpu.0/devfreq/17000000.gv11b/max_freq'], 
                                      capture_output=True, text=True)
                if gpu_freq.returncode == 0:
                    freq_mhz = int(gpu_freq.stdout.strip()) / 1000000
                    print(f"GPU Max Frequency: {freq_mhz}MHz")
            except:
                pass
            
            # 温度
            try:
                thermal = subprocess.run(['cat', '/sys/class/thermal/thermal_zone0/temp'], 
                                       capture_output=True, text=True)
                if thermal.returncode == 0:
                    temp = int(thermal.stdout.strip()) / 1000
                    print(f"CPU Temperature: {temp:.1f}°C")
            except:
                pass
                
        except Exception as e:
            print(f"System status check error: {e}")

# 应用AGX Orin优化
optimizer = AGXOrinOptimizer()
optimizer.maximize_performance()
optimizer.check_system_status()

# 配置参数
MODEL_DIR = "/home/seeed/yolov26_jetson/ultralytics_data"
TARGET_FPS = 30
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1536
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
OPTIMIZED_IMGSZ = 320

# 模型路径
DETECTION_ENGINE = os.path.join(MODEL_DIR, "yolo26n.engine")
POSE_ENGINE = os.path.join(MODEL_DIR, "yolo26n-pose.engine")
SEG_ENGINE = os.path.join(MODEL_DIR, "yolo26n-seg.engine")
DETECTION_MODEL = os.path.join(MODEL_DIR, "yolo26n.pt")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo26n-pose.pt")
SEG_MODEL = os.path.join(MODEL_DIR, "yolo26n-seg.pt")

# AGX Orin优化处理间隔
DETECT_INTERVAL = 4
POSE_INTERVAL = 10
SEG_INTERVAL = 15

CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class AGXOrinCameraCapture:
    """AGX Orin专用相机捕获器"""
    
    def __init__(self, camera_id, name):
        self.camera_id = camera_id
        self.name = name
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)  # AGX Orin可以处理更大队列
        self.capture_thread = None
        
        # 性能监控
        self.hardware_fps = 0
        self.frame_times = []
        self.last_frame_time = 0
        
    def start_capture(self):
        print(f"🎯 Starting AGX Orin capture for {self.name}")
        
        # AGX Orin优化的GStreamer管道
        gst_pipeline = self._build_agx_orin_pipeline()
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print(f"GStreamer failed for {self.name}, trying V4L2...")
            return self._start_v4l2_capture()
        
        print(f"✅ GStreamer pipeline active for {self.name}")
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def _build_agx_orin_pipeline(self):
        """构建AGX Orin优化的GStreamer管道"""
        pipeline = (
            f"v4l2src device=/dev/video{self.camera_id} "
            f"io-mode=dmabuf "  # DMA缓冲区，零拷贝
            f"do-timestamp=true "
            f"! "
            f"video/x-raw,format=YUY2,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},framerate={TARGET_FPS}/1 "
            f"! "
            f"queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! "
            f"nvvidconv "  # 硬件加速转换
            f"! "
            f"video/x-raw,width={PROCESS_WIDTH},height={PROCESS_HEIGHT},format=BGRx "
            f"! "
            f"videoconvert "
            f"! "
            f"video/x-raw,format=BGR "
            f"! "
            f"appsink max-buffers=2 drop=true sync=false"
        )
        
        print(f"AGX Orin pipeline for {self.name}: DMA+HW加速")
        return pipeline
    
    def _start_v4l2_capture(self):
        """V4L2降级方案（AGX Orin优化）"""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            return False
        
        # AGX Orin优化的V4L2设置
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # AGX Orin可以处理更多缓冲区
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        
        print(f"V4L2 capture active for {self.name} (AGX Orin optimized)")
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """AGX Orin优化的捕获循环"""
        target_frame_time = 1.0 / TARGET_FPS
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            start_time = time.time()
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # AGX Orin可以处理更快的帧率
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # 丢弃旧帧
                        except queue.Empty:
                            pass
                    
                    try:
                        self.frame_queue.put_nowait(frame)
                        frame_count += 1
                        
                        # 计算硬件FPS
                        current_time = time.time()
                        if current_time - last_time >= 1.0:
                            self.hardware_fps = frame_count / (current_time - last_time)
                            frame_count = 0
                            last_time = current_time
                            
                    except queue.Full:
                        pass
                else:
                    time.sleep(0.001)  # 更短的等待时间
                    
            except Exception as e:
                print(f"{self.name} capture error: {e}")
                time.sleep(0.01)
            
            # AGX Orin可以处理更紧的时序
            elapsed = time.time() - start_time
            if elapsed < target_frame_time * 0.8:  # 80%时间限制
                time.sleep(target_frame_time * 0.8 - elapsed)
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_hardware_fps(self):
        return self.hardware_fps
    
    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
        print(f"✅ {self.name} stopped")

class AGXOrinModelManager:
    """AGX Orin专用模型管理器"""
    
    def __init__(self):
        self.detection_model = None
        self.pose_model = None
        self.seg_model = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
    def load_models(self):
        try:
            from ultralytics import YOLO
            
            print(f"Loading models on {self.device}...")
            
            det_path = DETECTION_ENGINE if os.path.exists(DETECTION_ENGINE) else DETECTION_MODEL
            self.detection_model = YOLO(det_path)
            if det_path.endswith(".pt"):
                self.detection_model.fuse()
            print(f"✅ Detection model loaded: {det_path}")
            
            pose_path = POSE_ENGINE if os.path.exists(POSE_ENGINE) else POSE_MODEL
            self.pose_model = YOLO(pose_path)
            if pose_path.endswith(".pt"):
                self.pose_model.fuse()
            print(f"✅ Pose model loaded: {pose_path}")
            
            seg_path = SEG_ENGINE if os.path.exists(SEG_ENGINE) else SEG_MODEL
            self.seg_model = YOLO(seg_path)
            if seg_path.endswith(".pt"):
                self.seg_model.fuse()
            print(f"✅ Segmentation model loaded: {seg_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def get_detection_model(self):
        return self.detection_model
    
    def get_pose_model(self):
        return self.pose_model
    
    def get_seg_model(self):
        return self.seg_model

def agx_orin_postprocess_detection(detection_results, frame):
    """AGX Orin优化的检测后处理"""
    result_frame = frame.copy()
    
    for result in detection_results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            if hasattr(boxes, 'cpu'):
                boxes_cpu = boxes.cpu()
                xyxy = boxes_cpu.xyxy.numpy().astype(int)
                confs = boxes_cpu.conf.numpy()
                clss = boxes_cpu.cls.numpy().astype(int)
                
                # AGX Orin可以处理更多检测
                for i in range(min(len(xyxy), 3)):
                    x1, y1, x2, y2 = xyxy[i]
                    conf = confs[i]
                    cls = clss[i]
                    
                    if conf > 0.6:
                        color = COLORS[int(cls) % len(COLORS)]
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        if conf > 0.7:
                            names = getattr(result, 'names', None)
                            if isinstance(names, dict) and int(cls) in names:
                                label_name = names[int(cls)]
                            elif int(cls) < len(CLASSES):
                                label_name = CLASSES[int(cls)]
                            else:
                                label_name = str(int(cls))
                            label = f"{label_name}:{conf:.2f}"
                            cv2.putText(result_frame, label, (x1, max(12, y1-6)),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_frame

def agx_orin_postprocess_pose(pose_results, frame):
    """AGX Orin优化的姿态估计后处理"""
    result_frame = frame.copy()
    pose_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (0, 5), (0, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    for result in pose_results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints
            if hasattr(keypoints, 'cpu'):
                keypoints = keypoints.cpu()
                kpts = keypoints.xy.numpy()
                confs = keypoints.conf.numpy()
                
                if len(kpts) > 0 and len(kpts[0]) > 0:
                    for i, (x, y) in enumerate(kpts[0]):
                        if confs[0][i] > 0.5:
                            cv2.circle(result_frame, (int(x), int(y)), 3, (0, 255, 255), -1)
                    for idx1, idx2 in pose_connections:
                        if idx1 < len(kpts[0]) and idx2 < len(kpts[0]):
                            if confs[0][idx1] > 0.5 and confs[0][idx2] > 0.5:
                                pt1 = (int(kpts[0][idx1][0]), int(kpts[0][idx1][1]))
                                pt2 = (int(kpts[0][idx2][0]), int(kpts[0][idx2][1]))
                                cv2.line(result_frame, pt1, pt2, (0, 255, 0), 1)
    
    return result_frame

def agx_orin_postprocess_seg(seg_results, frame):
    """AGX Orin优化的分割后处理"""
    result_frame = frame.copy()
    
    for result in seg_results:
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks
            if hasattr(masks, 'cpu'):
                masks = masks.cpu()
                if hasattr(masks, 'data'):
                    mask_data = masks.data.numpy()
                    if len(mask_data.shape) == 3:
                        for i, mask in enumerate(mask_data[:2]):
                            color = COLORS[i % len(COLORS)]
                            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                            mask_resized = (mask_resized > 0.6).astype(np.uint8)
                            
                            colored_mask = np.zeros_like(result_frame)
                            colored_mask[mask_resized == 1] = color
                            
                            alpha = 0.35
                            result_frame = cv2.addWeighted(result_frame, 1, colored_mask, alpha, 0)
    
    return result_frame

class AGXOrinProcessor:
    """AGX Orin专用处理器"""
    
    def __init__(self, camera_id, name, task_type, model_manager):
        self.camera_id = camera_id
        self.name = name
        self.task_type = task_type
        self.capture = AGXOrinCameraCapture(camera_id, name)
        self.model_manager = model_manager
        self.running = False
        
        # AGX Orin性能参数
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.frame_id = 0
        self.last_detection = None
        self.last_secondary = None
        self.latest_result = None
        self.result_lock = threading.Lock()
        self.process_thread = None
        
        # 性能监控
        self.target_fps = TARGET_FPS
        self.process_times = []
        self.print_counter = 0
    
    def start(self):
        self.capture.start_capture()
        # 确保模型已加载
        if self.model_manager.get_detection_model() is None:
            try:
                from ultralytics import YOLO  # 避免延迟导入问题
                self.model_manager.load_models()
            except Exception as e:
                print(f"Model manager load error: {e}")
                return
        
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        print(f"🚀 AGX Orin processor started: {self.name} (Task: {self.task_type})")
    
    def _process_loop(self):
        """AGX Orin优化的处理循环"""
        while self.running:
            frame = self.capture.get_frame()
            if frame is not None:
                start_time = time.time()
                result = self._process_frame(frame)
                process_time = (time.time() - start_time) * 1000
                
                with self.result_lock:
                    self.latest_result = result
                
                # 记录处理时间
                self.process_times.append(process_time)
                if len(self.process_times) > 30:
                    self.process_times.pop(0)
                
                self._update_fps()
            else:
                time.sleep(0.001)  # AGX Orin可以更短等待
    
    def _process_frame(self, frame):
        """AGX Orin优化的单帧处理"""
        self.frame_id += 1
        process_start = time.time()
        
        process_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # 模型推理：AGX Orin可以并行处理
        model_start = time.time()
        
        # 检测模型：按间隔处理
        detection_model = self.model_manager.get_detection_model()
        if self.frame_id % DETECT_INTERVAL == 0:
            prediction = detection_model.predict(
                source=process_frame,
                task="detect",
                imgsz=OPTIMIZED_IMGSZ,
                device=0,
                half=True,
                verbose=False,
                conf=0.6,
                iou=0.45,
                max_det=3
            )
            detection_results = list(prediction)
            self.last_detection = detection_results
        else:
            detection_results = self.last_detection if self.last_detection is not None else []
        
        # 次要任务：AGX Orin可以更频繁处理
        secondary_results = []
        if self.task_type == 'pose' and self.frame_id % POSE_INTERVAL == 0:
            pose_model = self.model_manager.get_pose_model()
            try:
                prediction = pose_model.predict(
                    source=process_frame,
                    task="pose",
                    imgsz=OPTIMIZED_IMGSZ,
                    device=0,
                    half=True,
                    verbose=False,
                    conf=0.6,
                    iou=0.45,
                    max_det=2
                )
                secondary_results = list(prediction)
                self.last_secondary = secondary_results
            except Exception as e:
                print(f"{self.name} Pose error: {e}")
                secondary_results = []
        elif self.task_type == 'seg' and self.frame_id % SEG_INTERVAL == 0:
            seg_model = self.model_manager.get_seg_model()
            try:
                prediction = seg_model.predict(
                    source=process_frame,
                    task="segment",
                    imgsz=OPTIMIZED_IMGSZ,
                    device=0,
                    half=True,
                    verbose=False,
                    conf=0.55,
                    iou=0.45,
                    max_det=2
                )
                secondary_results = list(prediction)
                self.last_secondary = secondary_results
            except Exception as e:
                print(f"{self.name} Seg error: {e}")
                secondary_results = []
        else:
            # 使用缓存结果
            secondary_results = self.last_secondary if self.last_secondary is not None else []
        
        model_time = (time.time() - model_start) * 1000
        
        # 后处理绘制顺序：seg优先，detect覆盖其上
        if self.task_type == 'seg' and secondary_results:
            result_frame = agx_orin_postprocess_seg(secondary_results, process_frame)
            result_frame = agx_orin_postprocess_detection(detection_results, result_frame)
        else:
            result_frame = agx_orin_postprocess_detection(detection_results, process_frame)
            if self.task_type == 'pose' and secondary_results:
                result_frame = agx_orin_postprocess_pose(secondary_results, result_frame)
        
        # 缩放回显示尺寸
        display_frame = cv2.resize(result_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        # AGX Orin优化的状态信息
        avg_process_time = np.mean(self.process_times) if self.process_times else 0
        task_info = f"FPS: {self.fps:.1f} | Proc: {avg_process_time:.1f}ms"
        
        if self.task_type == 'pose':
            task_info += f" | Pose: {'Active' if self.frame_id % POSE_INTERVAL == 0 else 'Cache'}"
        elif self.task_type == 'seg':
            task_info += f" | Seg: {'Active' if self.frame_id % SEG_INTERVAL == 0 else 'Cache'}"
        
        cv2.putText(display_frame, task_info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        total_time = (time.time() - process_start) * 1000
        
        # AGX Orin性能报告（更频繁）
        self.print_counter += 1
        if self.print_counter >= 20:  # 每20帧（约0.67秒）
            print(f"🚀 {self.name} - Total: {total_time:.1f}ms, Model: {model_time:.1f}ms, "
                  f"Task: {self.task_type}, FPS: {self.fps:.1f}, Target: {self.target_fps}")
            self.print_counter = 0
        
        return display_frame
    
    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.last_time = current_time
            self.frame_count = 0
    
    def get_result(self):
        with self.result_lock:
            return self.latest_result.copy() if self.latest_result is not None else None
    
    def stop(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=1)
        if self.capture:
            self.capture.stop()
        print(f"✅ AGX Orin processor stopped: {self.name}")

class AGXOrinDualSystem:
    """AGX Orin双相机系统"""
    
    def __init__(self):
        self.camera1 = None
        self.camera2 = None
        self.camera1_available = False
        self.camera2_available = False
        self.running = False
    
    def find_cameras(self):
        """AGX Orin优化的相机发现"""
        print("🔍 Finding cameras for AGX Orin 30fps optimization...")
        
        available_cameras = []
        for idx in range(8):
            try:
                # 优先使用GStreamer测试
                test_pipeline = (
                    f"v4l2src device=/dev/video{idx} "
                    f"! video/x-raw,format=YUY2,width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},framerate={TARGET_FPS}/1 "
                    f"! appsink"
                )
                cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(idx)
                        print(f"✅ GMSL Camera {idx}: {frame.shape} @ {TARGET_FPS}fps (GStreamer)")
                    cap.release()
                else:
                    # 降级到V4L2
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            available_cameras.append(idx)
                            print(f"✅ GMSL Camera {idx}: {frame.shape} @ {actual_fps}fps (V4L2)")
                        cap.release()
                        
            except Exception as e:
                print(f"❌ Camera {idx}: {e}")
                continue
        
        print(f"\n🎯 Found {len(available_cameras)} cameras for AGX Orin: {available_cameras}")
        return available_cameras
    
    def start(self):
        print("🚀 AGX Orin Dual GMSL Camera System Startup")
        print(f"🎯 Target: {TARGET_FPS}fps on AGX Orin")
        print(f"📊 Process: {PROCESS_WIDTH}x{PROCESS_HEIGHT} (AGX Orin optimized)")
        print(f"🖥️  Display: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
        print("")
        
        available_cameras = self.find_cameras()
        
        if len(available_cameras) < 2:
            print("⚠️  Not enough cameras found, trying [0, 1]")
            available_cameras = [0, 1]
        
        # 加载模型
        model_manager = AGXOrinModelManager()
        if not model_manager.load_models():
            print("❌ Model loading failed")
            return
        
        try:
            self.camera1 = AGXOrinProcessor(available_cameras[0], "GMSL Camera 1", "pose", model_manager)
            self.camera1.start()
            self.camera1_available = True
            print("✅ GMSL Camera 1 started successfully (30fps + Pose)")
        except Exception as e:
            print(f"❌ GMSL Camera 1 startup failed: {e}")
        
        try:
            self.camera2 = AGXOrinProcessor(available_cameras[1], "GMSL Camera 2", "seg", model_manager)
            self.camera2.start()
            self.camera2_available = True
            print("✅ GMSL Camera 2 started successfully (30fps + Seg)")
        except Exception as e:
            print(f"❌ GMSL Camera 2 startup failed: {e}")
        
        if not self.camera1_available and not self.camera2_available:
            print("❌ No available GMSL cameras")
            return
        
        self.running = True
        print("")
        print("🎯 AGX Orin dual camera system ready for 30fps!")
        print("Press Ctrl+C to exit")
        print("")
        
        self.display_loop()
        self.stop()
    
    def display_loop(self):
        """AGX Orin优化的显示循环"""
        print("🖥️  Starting AGX Orin display...")
        
        pipeline1 = None
        pipeline2 = None
        appsrc1 = None
        appsrc2 = None
        
        try:
            if self.camera1_available:
                pipeline1 = Gst.parse_launch(
                    f"appsrc name=src ! "
                    f"video/x-raw, format=BGR, width={DISPLAY_WIDTH}, height={DISPLAY_HEIGHT}, framerate={TARGET_FPS}/1 ! "
                    f"videoconvert ! "
                    f"fpsdisplaysink video-sink=nveglglessink text-overlay=false sync=false"
                )
                appsrc1 = pipeline1.get_by_name("src")
                pipeline1.set_state(Gst.State.PLAYING)
                print("✅ Camera 1 display started (with FPS monitor)")
            
            if self.camera2_available:
                pipeline2 = Gst.parse_launch(
                    f"appsrc name=src ! "
                    f"video/x-raw, format=BGR, width={DISPLAY_WIDTH}, height={DISPLAY_HEIGHT}, framerate={TARGET_FPS}/1 ! "
                    f"videoconvert ! "
                    f"fpsdisplaysink video-sink=nveglglessink text-overlay=false sync=false"
                )
                appsrc2 = pipeline2.get_by_name("src")
                pipeline2.set_state(Gst.State.PLAYING)
                print("✅ Camera 2 display started (with FPS monitor)")
            
            loop = GLib.MainLoop()
            
            def push_frames():
                frame_count = 0
                last_time = time.time()
                disp_fps_1 = 0.0
                disp_last_1 = time.time()
                disp_frames_1 = 0
                disp_fps_2 = 0.0
                disp_last_2 = time.time()
                disp_frames_2 = 0
                
                while self.running:
                    current_time = time.time()
                    
                    # Camera 1
                    if self.camera1_available and appsrc1:
                        frame1 = self.camera1.get_result()
                        if frame1 is not None:
                            disp_frames_1 += 1
                            if current_time - disp_last_1 >= 1.0:
                                disp_fps_1 = disp_frames_1 / (current_time - disp_last_1)
                                disp_frames_1 = 0
                                disp_last_1 = current_time
                            # 叠加显示FPS与硬件FPS
                            hw_fps_1 = self.camera1.capture.get_hardware_fps()
                            cv2.putText(frame1, f"Disp:{disp_fps_1:.1f}  HW:{hw_fps_1:.1f}", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                            data = frame1.tobytes()
                            buffer = Gst.Buffer.new_wrapped(data)
                            appsrc1.emit("push-buffer", buffer)
                    
                    # Camera 2
                    if self.camera2_available and appsrc2:
                        frame2 = self.camera2.get_result()
                        if frame2 is not None:
                            disp_frames_2 += 1
                            if current_time - disp_last_2 >= 1.0:
                                disp_fps_2 = disp_frames_2 / (current_time - disp_last_2)
                                disp_frames_2 = 0
                                disp_last_2 = current_time
                            hw_fps_2 = self.camera2.capture.get_hardware_fps()
                            cv2.putText(frame2, f"Disp:{disp_fps_2:.1f}  HW:{hw_fps_2:.1f}", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                            data = frame2.tobytes()
                            buffer = Gst.Buffer.new_wrapped(data)
                            appsrc2.emit("push-buffer", buffer)
                    
                    # 性能报告
                    frame_count += 1
                    if current_time - last_time >= 5.0:
                        effective_fps = frame_count / (current_time - last_time)
                        print(f"📊 System display FPS: {effective_fps:.1f} (target: {TARGET_FPS * 2})")
                        frame_count = 0
                        last_time = current_time
                    
                    # AGX Orin可以更精确控制
                    time.sleep(1.0 / (TARGET_FPS * 2.0))
                
                loop.quit()
            
            thread = threading.Thread(target=push_frames, daemon=True)
            thread.start()
            
            loop.run()
                
        except KeyboardInterrupt:
            print("\n⏹️  User interruption")
        except Exception as e:
            print(f"Display error: {e}")
        finally:
            if pipeline1:
                pipeline1.set_state(Gst.State.NULL)
            if pipeline2:
                pipeline2.set_state(Gst.State.NULL)
            print("✅ Display resources released")
    
    def stop(self):
        print("Stopping AGX Orin dual camera system")
        self.running = False
        
        if self.camera1_available:
            self.camera1.stop()
        if self.camera2_available:
            self.camera2.stop()
        
        print("✅ AGX Orin system stopped")

def main():
    try:
        system = AGXOrinDualSystem()
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
PYEOF
