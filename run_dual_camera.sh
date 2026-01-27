#!/bin/bash

# 清理现有容器
docker rm -f dual-camera-system 2>/dev/null

# 清理可能存在的dual_camera_system.py文件
rm -f dual_camera_system.py 2>/dev/null

# 配置
MODEL_DIR="/home/seeed/ultralytics_data"
# 性能优化配置
OPTIMIZED_IMGSZ=640  # TensorRT引擎固定尺寸，保持640
JPEG_QUALITY=60  # 降低JPEG质量以减少网络传输时间
CAMERA_WIDTH=320  # 相机宽度
CAMERA_HEIGHT=240  # 相机高度
CAMERA_FPS=30  # 相机帧率

# 混合量化配置 - FP16用于检测，INT8用于姿态估计和分割
USE_MIXED_QUANTIZATION=true
FP16_MODELS="yolo26n"  # 检测模型使用FP16
INT8_MODELS="yolo26n-pose yolo26n-seg"  # 姿态和分割模型使用INT8
BATCH_SIZE=1  # 批处理大小优化
MAX_DETECTIONS=3  # 最大检测数量限制

# Jetson 性能优化建议
echo "=== Jetson 性能优化提示 ==="
echo "建议运行以下命令以获得最佳性能："
echo "  sudo nvpmodel -m 0  # 最大性能模式"
echo "  sudo jetson_clocks  # 启用最大时钟频率"
echo "  tegrastats  # 监控系统性能"
echo ""
echo "=== TensorRT FP16/INT8 混合量化优化提示 ==="
echo "检查TensorRT引擎是否使用混合量化："
echo "  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n.engine --fp16"
echo "  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n-pose.engine --int8"
echo ""
echo "如果需要重新导出混合量化引擎："
echo "  # FP16检测模型 (高精度检测)"
echo "  yolo export model=yolo26n.pt format=engine device=0 half=True"
echo "  # INT8姿态模型 (快速推理)"  
echo "  yolo export model=yolo26n-pose.pt format=engine device=0 int8=True"
echo "  # INT8分割模型 (快速分割)"
echo "  yolo export model=yolo26n-seg.pt format=engine device=0 int8=True"
echo ""

echo "=== 双USB相机图像处理系统启动脚本 ==="
echo "基于YOLOv26模型和TensorRT加速"
echo ""
echo "系统配置:"
echo "- 相机1: 目标检测 + 姿态估计"
echo "- 相机2: 目标检测 + SAM模型"
echo "- Web服务器: http://localhost:5000"
echo ""
echo "模型目录: $MODEL_DIR"
echo ""
echo "按 Ctrl+C 退出系统"
echo ""

# 检查模型文件
echo "=== 检查模型文件 ==="
if [ ! -f "$MODEL_DIR/yolo26n.engine" ] || [ ! -f "$MODEL_DIR/yolo26n-pose.engine" ]; then
    echo "错误: 必要的模型文件不存在"
    exit 1
fi

echo "✅ 模型文件检查完成"
echo ""

# 创建Python脚本文件
cat > dual_camera_system.py << 'PYTHONEOF'
#!/usr/bin/env python3
"""
双USB相机图像处理系统
基于YOLOv26模型和TensorRT加速
"""

import cv2
import numpy as np
import os
import time
from threading import Thread
import queue

# GPU优化函数
def optimize_gpu_memory():
    """优化GPU内存使用"""
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
    """获取推理统计信息"""
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

# 配置
MODEL_DIR = "/models"
DETECTION_MODEL = os.path.join(MODEL_DIR, "yolo26n.engine")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo26n-pose.engine")
SAM_MODEL = os.path.join(MODEL_DIR, "yolo26n-seg.engine")

# 性能优化配置
OPTIMIZED_IMGSZ = 640  # TensorRT引擎固定尺寸，保持640
JPEG_QUALITY = 60  # 降低JPEG质量以减少网络传输时间
CAMERA_WIDTH = 320  # 相机宽度
CAMERA_HEIGHT = 240  # 相机高度
CAMERA_FPS = 30  # 相机帧率

# 混合量化配置
USE_MIXED_QUANTIZATION = True
FP16_MODELS = ["yolo26n"]  # 检测模型使用FP16
INT8_MODELS = ["yolo26n-pose", "yolo26n-seg"]  # 姿态和分割模型使用INT8
BATCH_SIZE = 1  # 批处理大小优化
MAX_DETECTIONS = 3  # 最大检测数量限制

# 姿态关键点阈值
POSE_KPT_THRESHOLD = 0.2

# COCO 姿态关键点连线（17 点）
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]

# 类别列表（检测用）
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

# 自定义后处理函数
def custom_postprocess_pose(detection_results, pose_results, frame):
    """自定义姿态后处理函数"""
    # 创建结果帧的副本
    result_frame = frame.copy()
    
    # 先处理目标检测结果
    for result in detection_results:
        # 获取框
        boxes = result.boxes
        if boxes is not None:
            # 一次性取完所有数据，减少 CPU-GPU 同步
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            
            # 遍历每个框
            for i in range(len(xyxy)):
                # 获取框坐标
                x1, y1, x2, y2 = xyxy[i]
                # 获取置信度
                conf = confs[i]
                # 获取类别
                cls = clss[i]
                
                # 绘制框
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制类别和置信度
                label = f"{CLASSES[cls]}: {conf:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 再处理姿态估计结果
    for result in pose_results:
        # 获取关键点
        keypoints = result.keypoints
        if keypoints is not None:
            # 一次性取完所有关键点数据，减少 CPU-GPU 同步
            kpts = keypoints.data.cpu().numpy()
            
            # 遍历每个人
            for i in range(kpts.shape[0]):
                # 获取关键点坐标
                pts = kpts[i, :, :2].astype(int)
                # 获取置信度
                confs = kpts[i, :, 2]
                
                # 画点
                for j, (x, y) in enumerate(pts):
                    if confs[j] > POSE_KPT_THRESHOLD:
                        cv2.circle(result_frame, (x, y), 3, (0, 255, 255), -1)
                
                # 画连线
                for j, k in SKELETON:
                    if confs[j] > POSE_KPT_THRESHOLD and confs[k] > POSE_KPT_THRESHOLD:
                        cv2.line(result_frame, tuple(pts[j]), tuple(pts[k]), (0, 200, 0), 2)
    
    return result_frame

def get_optimal_model_config(model_name):
    """获取最优模型配置，包括量化策略"""
    config = {
        'device': 0,  # GPU设备
        'imgsz': OPTIMIZED_IMGSZ,
        'batch': BATCH_SIZE,
        'conf': 0.25,  # 置信度阈值
        'iou': 0.45,   # IoU阈值
        'max_det': MAX_DETECTIONS,
        'half': False,  # 默认不使用FP16
        'int8': False,  # 默认不使用INT8
        'optimize': True,  # 启用优化
        'stream': True,   # 启用流模式
        'agnostic_nms': False,  # 类无关NMS
        'classes': [0] if 'pose' in model_name or 'seg' in model_name else None  # 姿态和分割只检测person
    }
    
    # 根据模型类型设置量化策略
    if USE_MIXED_QUANTIZATION:
        if any(fp16_model in model_name for fp16_model in FP16_MODELS):
            config['half'] = True  # FP16量化
            config['int8'] = False
        elif any(int8_model in model_name for int8_model in INT8_MODELS):
            config['half'] = False
            config['int8'] = True  # INT8量化
    
    return config

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, name):
        self.name = name
        self.frame_times = []
        self.inference_times = []
        self.fps_history = []
        self.target_fps = CAMERA_FPS
        self.window_size = 30
    
    def add_frame_time(self, frame_time):
        """添加帧处理时间"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def add_inference_time(self, inference_time):
        """添加推理时间"""
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)
    
    def add_fps(self, fps):
        """添加FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
    
    def get_stats(self):
        """获取性能统计"""
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
        """打印性能统计"""
        stats = self.get_stats()
        if stats:
            print(f"[{self.name}] 性能统计 - FPS: {stats['avg_fps']:.1f}/{stats['fps_target']}, "
                  f"帧时间: {stats['avg_frame_time_ms']:.1f}ms, "
                  f"推理时间: {stats['avg_inference_time_ms']:.1f}ms, "
                  f"性能比: {stats['performance_ratio']*100:.1f}%")

def custom_postprocess_sam(detection_results, sam_results, frame):
    """自定义SAM后处理函数"""
    # 创建结果帧的副本
    result_frame = frame.copy()
    
    # 先处理目标检测结果
    for result in detection_results:
        # 获取框
        boxes = result.boxes
        if boxes is not None:
            # 一次性取完所有数据，减少 CPU-GPU 同步
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            
            # 遍历每个框
            for i in range(len(xyxy)):
                # 获取框坐标
                x1, y1, x2, y2 = xyxy[i]
                # 获取置信度
                conf = confs[i]
                # 获取类别
                cls = clss[i]
                
                # 绘制框
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制类别和置信度
                label = f"{CLASSES[cls]}: {conf:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 再处理SAM模型结果
    for result in sam_results:
        # 获取分割
        masks = result.masks
        if masks is not None:
            # 一次性取完所有分割数据，减少 CPU-GPU 同步
            mask_data = masks.data.cpu().numpy()
            
            # 遍历每个分割
            for i in range(mask_data.shape[0]):
                # 获取掩码
                mask = mask_data[i]
                # 应用掩码
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                color = COLORS[i % len(COLORS)]
                result_frame[mask > 0.5] = result_frame[mask > 0.5] * 0.7 + color * 0.3
    
    return result_frame

class CameraProcessor:
    """相机处理器基类"""
    def __init__(self, camera_id, name):
        """初始化相机处理器"""
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
        self.latest_result = None  # 存储最新的处理结果
        self.infer_busy = False  # 推理忙标志，用于丢帧策略
        self.print_counter = 0  # 打印计数器，减少输出频率
        self.performance_monitor = PerformanceMonitor(name)  # 性能监控器
    
    def start_capture(self):
        """启动相机捕获"""
        print("=== 尝试打开相机 %d (%s) ===" % (self.camera_id, self.name))
        
        # 尝试使用GStreamer管道（针对Jetson优化）
        try:
            # 为不同的相机ID创建不同的GStreamer管道
            device_path = f"/dev/video{self.camera_id}"
            gst_str = (
                f"v4l2src device={device_path} ! "
                f"video/x-raw, width={CAMERA_WIDTH}, height={CAMERA_HEIGHT}, framerate={CAMERA_FPS}/1 ! "
                "videoconvert ! "
                "video/x-raw, format=BGR ! appsink drop=true sync=false"
            )
            
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            
            if self.cap.isOpened():
                print("✅ 相机 %d (%s) 打开成功 (GStreamer)" % (self.camera_id, self.name))
                print("相机参数: 320x240 @ 30fps")
                
                # 测试读取几帧
                for i in range(5):
                    time.sleep(0.3)
                    ret, frame = self.cap.read()
                    if ret:
                        print("✅ 相机 %d 测试读取成功" % self.camera_id)
                        return True
                    else:
                        print("⚠️ 相机 %d 测试读取失败" % self.camera_id)
                
                self.cap.release()
        except Exception as e:
            print("⚠️ GStreamer打开失败: %s" % e)
            if self.cap:
                self.cap.release()
        
        # 回退到标准V4L2
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if self.cap.isOpened():
                print("✅ 相机 %d (%s) 打开成功 (V4L2)" % (self.camera_id, self.name))
                
                # 优化相机参数
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 禁用自动曝光以减少延迟
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 禁用自动白平衡以减少延迟
                
                # 验证相机参数
                width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                print("相机参数: %dx%d @ %dfps" % (int(width), int(height), fps))
                
                # 测试读取几帧
                for i in range(5):
                    time.sleep(0.3)
                    ret, frame = self.cap.read()
                    if ret:
                        print("✅ 相机 %d 测试读取成功" % self.camera_id)
                        return True
                    else:
                        print("⚠️ 相机 %d 测试读取失败" % self.camera_id)
                
                self.cap.release()
        except Exception as e:
            print("⚠️ V4L2打开失败: %s" % (e))
            if self.cap:
                self.cap.release()
        
        # 所有尝试都失败
        raise RuntimeError("无法打开相机 %d" % self.camera_id)
    
    def capture_and_process_frames(self):
        """捕获并处理相机帧的线程"""
        while self.running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and self.model_loaded:
                        # 推理忙时直接丢帧，避免延迟爆炸
                        if self.infer_busy:
                            continue
                        # 直接处理帧，不使用队列
                        self.process_frame(frame)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print("相机 %d 处理错误: %s" % (self.camera_id, e))
                time.sleep(1)
    
    def process_frame(self, frame):
        """处理单帧"""
        # 设置推理忙标志
        self.infer_busy = True
        
        # 详细时间测量
        total_start = time.time()
        
        # 预处理时间（包含在模型调用中）
        model_start = time.time()
        # 使用TensorRT优化推理
        detection_results = self.detection_model(frame, imgsz=OPTIMIZED_IMGSZ, device=0)
        pose_results = self.secondary_model(frame, task='pose', imgsz=OPTIMIZED_IMGSZ, device=0)
        model_time = (time.time() - model_start) * 1000
        
        # 后处理时间
        postprocess_start = time.time()
        
        # 单独测量后处理方法的执行时间
        plot_start = time.time()
        # 使用自定义后处理函数替代 plot() 方法，同时传递目标检测结果
        pose_frame = custom_postprocess_pose(detection_results, pose_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(pose_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # 总时间
        total_time = (time.time() - total_start) * 1000
        
        # 更新FPS
        self.update_fps()
        
        # 减少打印频率，每30帧打印一次
        self.print_counter += 1
        if self.print_counter >= 30:
            print(f"相机1 - 总时间: {total_time:.1f}ms, 模型时间: {model_time:.1f}ms, 后处理时间: {postprocess_time:.1f}ms, plot时间: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # 存储最新结果
        self.latest_result = pose_frame
        
        # 清除推理忙标志
        self.infer_busy = False
    
    def load_detection_model(self):
        """加载目标检测模型"""
        try:
            from ultralytics import YOLO
            # 获取优化配置
            config = get_optimal_model_config(DETECTION_MODEL)
            self.detection_model = YOLO(DETECTION_MODEL)
            # 应用优化配置
            self.detection_model.model.fuse()  # 融合层以优化推理
            print(f"✅ 检测模型加载完成 - FP16: {config['half']}, INT8: {config['int8']}")
            return True
        except Exception as e:
            print("❌ 目标检测模型加载失败: %s" % e)
            return False
    
    def load_secondary_model(self):
        """加载次要模型（由子类实现）"""
        pass
    
    def load_models(self):
        """加载模型"""
        try:
            # 对于Camera1，Pose模型已经包含检测功能，不需要单独加载检测模型
            if self.name == "相机1":
                if not self.load_secondary_model():
                    return
            else:
                if not self.load_detection_model() or not self.load_secondary_model():
                    return
            
            self.model_loaded = True
            print("✅ %s 模型加载完成" % self.name)
        except Exception as e:
            print("❌ %s 模型加载失败: %s" % (self.name, e))
            self.model_loaded = False
    
    def start(self):
        """启动处理器"""
        self.running = True
        
        # 打开相机
        self.start_capture()
        
        # 加载模型
        self.load_models()
        
        # 启动捕获和处理线程
        Thread(target=self.capture_and_process_frames, daemon=True).start()
        
        print("✅ 相机 %d (%s) 处理线程启动完成" % (self.camera_id, self.name))
    
    def stop(self):
        """停止处理器"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        print("✅ 相机 %d (%s) 已停止" % (self.camera_id, self.name))
    
    def get_result(self):
        """获取处理结果"""
        return self.latest_result
    
    def get_result_nonblock(self):
        """非阻塞获取处理结果"""
        return self.latest_result
    
    def update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.last_time = current_time
            self.frame_count = 0

class Camera1Processor(CameraProcessor):
    """相机1处理器：目标检测 + 姿态估计"""
    def __init__(self):
        super().__init__(0, "相机1")
    
    def load_secondary_model(self):
        """加载姿态估计模型"""
        try:
            from ultralytics import YOLO
            # 获取优化配置
            config = get_optimal_model_config(POSE_MODEL)
            self.secondary_model = YOLO(POSE_MODEL)
            # 应用优化配置
            self.secondary_model.model.fuse()  # 融合层以优化推理
            # 不需要单独加载检测模型，Pose模型已经包含检测功能
            self.detection_model = None
            print(f"✅ 姿态模型加载完成 - FP16: {config['half']}, INT8: {config['int8']}")
            return True
        except Exception as e:
            print("❌ 姿态估计模型加载失败: %s" % e)
            return False
    
    def process_frame(self, frame):
        """处理单帧"""
        # 设置推理忙标志
        self.infer_busy = True
        
        # 详细时间测量
        total_start = time.time()
        
        # 预处理时间（包含在模型调用中）
        model_start = time.time()
        # 只运行一次Pose模型推理，Pose模型已经包含检测功能
        # 使用优化配置进行推理
        pose_config = get_optimal_model_config(POSE_MODEL)
        pose_results = self.secondary_model(frame, task='pose', **pose_config)
        model_time = (time.time() - model_start) * 1000
        
        # 后处理时间
        postprocess_start = time.time()
        
        # 单独测量后处理方法的执行时间
        plot_start = time.time()
        # 直接从pose_results中提取检测框和关键点进行绘制
        pose_frame = custom_postprocess_pose(pose_results, pose_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(pose_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # 总时间
        total_time = (time.time() - total_start) * 1000
        
        # 更新FPS
        self.update_fps()
        
        # 记录性能数据
        self.performance_monitor.add_frame_time(total_time)
        self.performance_monitor.add_inference_time(model_time)
        self.performance_monitor.add_fps(self.fps)
        
        # 定期优化GPU内存和打印性能统计
        if self.frame_count % 100 == 0:
            optimize_gpu_memory()
            stats = get_inference_stats()
            if stats:
                print(f"GPU内存使用: {stats['memory_used_gb']:.2f}GB / {stats['memory_reserved_gb']:.2f}GB")
            self.performance_monitor.print_stats()
        
        # 减少打印频率，每30帧打印一次
        self.print_counter += 1
        if self.print_counter >= 30:
            print(f"相机1 - 总时间: {total_time:.1f}ms, Pose模型: {pose_results[0].speed['inference']:.1f}ms, 后处理: {postprocess_time:.1f}ms, plot: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # 存储最新结果
        self.latest_result = pose_frame
        
        # 清除推理忙标志
        self.infer_busy = False

class Camera2Processor(CameraProcessor):
    """相机2处理器：目标检测 + SAM模型"""
    def __init__(self):
        super().__init__(1, "相机2")
        self.frame_id = 0  # 帧计数器，用于SAM低频触发
        self.last_sam = None  # 缓存上一次的SAM结果
    
    def load_secondary_model(self):
        """加载SAM模型"""
        try:
            from ultralytics import YOLO
            if os.path.exists(SAM_MODEL):
                # 获取优化配置
                config = get_optimal_model_config(SAM_MODEL)
                self.secondary_model = YOLO(SAM_MODEL)
                # 应用优化配置
                self.secondary_model.model.fuse()  # 融合层以优化推理
                print(f"✅ SAM模型加载完成 - FP16: {config['half']}, INT8: {config['int8']}")
            else:
                print("⚠️ SAM模型不存在，将使用目标检测的分割结果")
            return True
        except Exception as e:
            print("❌ SAM模型加载失败: %s" % e)
            return True
    
    def process_frame(self, frame):
        """处理单帧"""
        # 设置推理忙标志
        self.infer_busy = True
        
        # 详细时间测量
        total_start = time.time()
        
        # 模型时间（包含预处理和推理）
        model_start = time.time()
        
        # 获取优化配置
        detection_config = get_optimal_model_config(DETECTION_MODEL)
        
        # 先运行目标检测模型，每帧都运行
        detection_results = self.detection_model(frame, **detection_config)
        
        # SAM模型改为低频触发：每5帧运行一次
        self.frame_id += 1
        SAM_INTERVAL = 5  # SAM模型每N帧运行一次
        
        if self.secondary_model and self.frame_id % SAM_INTERVAL == 0:
            try:
                sam_config = get_optimal_model_config(SAM_MODEL)
                sam_results = self.secondary_model(frame, **sam_config)
                self.last_sam = sam_results  # 缓存SAM结果
            except Exception as e:
                print("SAM推理失败: %s" % e)
        
        # 使用缓存的SAM结果
        sam_results = self.last_sam if self.last_sam is not None else detection_results
        
        model_time = (time.time() - model_start) * 1000
        
        # 后处理时间
        postprocess_start = time.time()
        
        # 单独测量后处理方法的执行时间
        plot_start = time.time()
        # 使用自定义后处理函数替代 plot() 方法，同时传递目标检测结果
        result_frame = custom_postprocess_sam(detection_results, sam_results, frame)
        plot_time = (time.time() - plot_start) * 1000
        
        cv2.putText(result_frame, 'FPS: %.1f' % self.fps, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        postprocess_time = (time.time() - postprocess_start) * 1000
        
        # 总时间
        total_time = (time.time() - total_start) * 1000
        
        # 更新FPS
        self.update_fps()
        
        # 定期优化GPU内存
        if self.frame_count % 100 == 0:
            optimize_gpu_memory()
            stats = get_inference_stats()
            if stats:
                print(f"GPU内存使用: {stats['memory_used_gb']:.2f}GB / {stats['memory_reserved_gb']:.2f}GB")
        
        # 减少打印频率，每30帧打印一次
        self.print_counter += 1
        if self.print_counter >= 30:
            sam_status = "新" if self.frame_id % SAM_INTERVAL == 0 else "缓存"
            print(f"相机2 - 总时间: {total_time:.1f}ms, 检测模型: {detection_results[0].speed['inference']:.1f}ms, SAM: {sam_status}, 后处理: {postprocess_time:.1f}ms, plot: {plot_time:.1f}ms")
            self.print_counter = 0
        
        # 存储最新结果
        self.latest_result = result_frame
        
        # 清除推理忙标志
        self.infer_busy = False

class DualCameraSystem:
    """双相机系统"""
    def __init__(self):
        """初始化双相机系统"""
        self.camera1 = None
        self.camera2 = None
        self.camera1_available = False
        self.camera2_available = False
        self.running = False
    
    def find_available_cameras(self):
        """寻找可用的相机"""
        print("=== 寻找可用的相机 ===")
        
        available_cameras = []
        for idx in [0, 1, 2, 3, 4]:
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    time.sleep(0.3)
                    ret, frame = cap.read()
                    if ret:
                        print("✅ 相机索引 %d 可用" % idx)
                        available_cameras.append(idx)
                    cap.release()
                else:
                    cap.release()
            except Exception:
                pass
        
        print("找到 %d 个可用相机: %s" % (len(available_cameras), available_cameras))
        return available_cameras
    
    def start_web_server(self):
        """启动Web服务器"""
        try:
            from flask import Flask, Response
            import threading
            import time
            
            app = Flask(__name__)
            
            def generate_frames(camera_id):
                """生成视频流帧"""
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
                """主页"""
                return '''
                <html>
                <head>
                    <title>双相机图像处理系统</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #333; }
                        .camera-container { margin: 20px 0; }
                        h2 { color: #666; }
                        img { border: 1px solid #ddd; padding: 10px; }
                    </style>
                </head>
                <body>
                    <h1>双相机图像处理系统</h1>
                    <p>基于YOLOv26模型和TensorRT加速</p>
                    
                    <div class="camera-container">
                        <h2>相机1: 目标检测 + 姿态估计</h2>
                        <img src="/video_feed/1" width="640">
                    </div>
                    
                    <div class="camera-container">
                        <h2>相机2: 目标检测 + SAM模型</h2>
                        <img src="/video_feed/2" width="640">
                    </div>
                </body>
                </html>
                '''
            
            @app.route('/video_feed/<int:camera_id>')
            def video_feed(camera_id):
                """视频流"""
                return Response(generate_frames(camera_id),
                                mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # 启动Web服务器
            threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, 
                                                  debug=False, threaded=True), 
                          daemon=True).start()
            
            print("✅ Web服务器启动成功")
            print("请在浏览器中访问: http://localhost:5000")
            
        except Exception as e:
            print("❌ Web服务器启动失败: %s" % e)
    
    def start(self):
        """启动双相机系统"""
        print("=== 双USB相机图像处理系统启动 ===")
        print("基于YOLOv26模型和TensorRT加速")
        print("相机1: 目标检测 + 姿态估计")
        print("相机2: 目标检测 + SAM模型")
        print("")
        
        # 寻找可用的相机
        available_cameras = self.find_available_cameras()
        
        if len(available_cameras) == 0:
            print("❌ 没有可用的相机，系统无法启动")
            return
        
        # 启动相机1
        if len(available_cameras) >= 1:
            print("启动相机1...")
            try:
                self.camera1 = Camera1Processor()
                self.camera1.camera_id = available_cameras[0]
                self.camera1.start()
                print("✅ 相机1启动成功")
                self.camera1_available = True
            except Exception as e:
                print("❌ 相机1启动失败: %s" % e)
        
        # 启动相机2
        if len(available_cameras) >= 2:
            print("启动相机2...")
            try:
                self.camera2 = Camera2Processor()
                self.camera2.camera_id = available_cameras[1]
                self.camera2.start()
                print("✅ 相机2启动成功")
                self.camera2_available = True
            except Exception as e:
                print("❌ 相机2启动失败: %s" % e)
        else:
            print("⚠️ 没有足够的可用相机，只启动相机1")
        
        # 检查是否有可用相机
        if not self.camera1_available and not self.camera2_available:
            print("❌ 没有可用的相机，系统无法启动")
            return
        
        # 启动Web服务器
        self.start_web_server()
        
        self.running = True
        print("")
        print("=== 双相机系统启动完成 ===")
        print("可用相机: %s%s" % ("相机1 " if self.camera1_available else "", "相机2" if self.camera2_available else ""))
        print("按 Ctrl+C 退出系统")
        print("")
        
        # 主循环
        while self.running:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.running = False
                break
        
        # 停止系统
        self.stop()
    
    def stop(self):
        """停止双相机系统"""
        print("=== 正在停止双相机系统 ===")
        self.running = False
        
        if self.camera1_available:
            self.camera1.stop()
        if self.camera2_available:
            self.camera2.stop()
        
        print("=== 双相机系统已停止 ===")

def main():
    """主函数"""
    try:
        # 检查模型文件
        if not os.path.exists(DETECTION_MODEL) or not os.path.exists(POSE_MODEL):
            raise FileNotFoundError("必要的模型文件不存在")
        
        # 启动双相机系统
        system = DualCameraSystem()
        system.start()
        
    except Exception as e:
        print("错误: %s" % e)

if __name__ == "__main__":
    main()
PYTHONEOF

# 启动Docker容器
docker run --privileged -it --name dual-camera-system --runtime=nvidia --gpus all --ipc=host --network=host \
    -v "$MODEL_DIR:/models" \
    -v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py" \
    ultralytics/ultralytics:latest-jetson-jetpack6 \
    bash -c "pip install flask --quiet && python3 /app/dual_camera_system.py"

# 清理临时文件
rm -f dual_camera_system.py
