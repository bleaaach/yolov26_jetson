# yolov26_jetson
# YOLOv26双USB相机图像处理系统教程

## 系统介绍

本教程将从零开始，教您如何搭建一个基于YOLOv26模型和TensorRT加速的双USB相机图像处理系统。该系统具有以下功能：

- **双相机并行处理**：同时支持两个USB相机的实时视频流处理
- **多任务视觉分析**：目标检测、姿态估计、图像分割（SAM模型）
- **TensorRT加速**：使用NVIDIA TensorRT引擎大幅提升推理速度
- **Web界面预览**：通过浏览器实时查看处理结果
- **MJPEG压缩**：降低USB带宽占用，提高系统稳定性

## 一、环境准备

### 1. 硬件要求

- NVIDIA Jetson设备（Nano、Xavier NX、Orin等）
- JetPack 6.0或更高版本
- 2个USB相机（支持MJPEG格式）
- 足够的电源供应（建议使用5V 4A以上电源）

### 2. 软件依赖

系统已内置以下软件：
- Docker
- Python 3.8+
- OpenCV
- Flask
- Ultralytics YOLO库
- TensorRT

## 二、Docker环境配置

### 1. Docker的优势

本系统使用Docker容器化部署，具有以下优势：

- **环境隔离**：所有依赖项都封装在容器中，避免与主机系统冲突
- **快速部署**：一键启动，无需手动安装复杂的依赖库
- **GPU支持**：内置NVIDIA GPU驱动和TensorRT支持
- **可移植性**：在不同设备上保持相同的运行环境

### 2. 检查Docker状态

确保Docker服务正在运行：

```bash
sudo systemctl status docker
```

如果Docker未运行，启动它：

```bash
sudo systemctl start docker
```

### 3. Docker镜像信息

系统使用官方的Ultralytics Docker镜像：

```
ultralytics/ultralytics:latest-jetson-jetpack6
```

该镜像包含：
- Python 3.8+
- PyTorch和Torchvision
- OpenCV
- TensorRT
- Ultralytics YOLO库
- 所有必要的依赖项

## 三、模型下载与准备

### 1. 模型文件目录

模型文件存储在 `/home/seeed/ultralytics_data` 目录中。如果该目录不存在，请先创建：

```bash
mkdir -p /home/seeed/ultralytics_data
cd /home/seeed/ultralytics_data
```

### 2. 下载YOLOv26模型

使用以下命令下载YOLOv26系列模型：

```bash
# 下载目标检测模型
yolo export model=yolov26n.pt format=engine device=0

# 下载姿态估计模型
yolo export model=yolov26n-pose.pt format=engine device=0

# 下载分割模型（SAM）
yolo export model=yolov26n-seg.pt format=engine device=0
```

### 3. 验证模型文件

下载完成后，确认模型文件存在：

```bash
ls -la /home/seeed/ultralytics_data/
```

您应该看到以下文件：
- `yolo26n.engine`（目标检测）
- `yolo26n-pose.engine`（姿态估计）
- `yolo26n-seg.engine`（图像分割）

## 三、系统配置

### 1. 检查USB相机

连接两个USB相机，然后检查相机设备：

```bash
ls -la /dev/video*
```

系统应该识别到两个或更多的视频设备（如 `/dev/video0`、`/dev/video1` 等）。

### 2. 启动脚本配置

启动脚本已配置好以下参数：

- **相机参数**：320x240（相机1）、640x360（相机2）
- **帧率**：30fps（相机1）、25fps（相机2）
- **压缩格式**：MJPEG（两个相机均启用）
- **Web服务器**：http://localhost:5000
- **多线程**：已启用，支持并发请求

## 四、启动系统

### 1. 运行启动脚本

```bash
cd /home/seeed
./run_dual_camera_final_fixed.sh
```

### 2. 系统启动流程

启动脚本会执行以下步骤：

1. **清理现有容器**：移除之前的dual-camera-system容器
2. **检查模型文件**：验证所有必要的模型文件是否存在
3. **创建Python脚本**：生成包含完整系统代码的Python文件
4. **启动Docker容器**：配置GPU支持、USB设备映射、网络设置等
5. **检测可用相机**：自动识别系统中的USB相机
6. **初始化相机**：启用MJPEG压缩，设置分辨率和帧率
7. **加载模型**：加载YOLOv26模型到内存
8. **启动Web服务器**：在容器中启动Flask服务器

### 3. Docker容器配置详情

启动脚本使用以下Docker命令启动容器：

```bash
docker run --privileged -it --name dual-camera-system --runtime=nvidia --gpus all --ipc=host --network=host \
    -e DISPLAY=:1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$MODEL_DIR:/models" \
    -v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py" \
    ultralytics/ultralytics:latest-jetson-jetpack6 \
    bash -c "pip install flask --quiet && python3 /app/dual_camera_system.py"
```

**参数说明**：

- `--privileged`：授予容器特权，允许访问USB设备
- `--runtime=nvidia`：使用NVIDIA容器运行时
- `--gpus all`：启用所有可用的GPU
- `--ipc=host`：使用主机的IPC命名空间，提高性能
- `--network=host`：使用主机网络，便于Web服务器访问
- `-v "$MODEL_DIR:/models"`：映射模型文件目录到容器
- `-v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py"`：映射Python脚本到容器

### 4. 验证系统启动

查看启动日志，确认：

- ✅ Docker容器启动成功
- ✅ 两个相机都已成功初始化
- ✅ 模型加载成功
- ✅ Web服务器启动成功
- ✅ 视频流正常传输

### 5. Docker容器管理

#### 查看运行中的容器

```bash
docker ps
```

#### 查看容器日志

```bash
# 查看所有日志
docker logs dual-camera-system

# 实时查看日志（按Ctrl+C退出）
docker logs -f dual-camera-system
```

#### 停止容器

```bash
docker stop dual-camera-system
```

#### 删除容器

```bash
docker rm dual-camera-system
```

#### 重启系统

如果需要重启系统，使用以下命令：

```bash
# 停止并删除现有容器
docker rm -f dual-camera-system

# 重新启动系统
cd /home/seeed
./run_dual_camera_final_fixed.sh
```

## 五、使用系统

### 1. 访问Web界面

在浏览器中打开：

```
http://localhost:5000
```

您将看到两个相机的实时视频流，包括：
- 相机1：目标检测 + 姿态估计
- 相机2：目标检测 + SAM模型分割

### 2. 系统功能说明

#### 相机1（左侧）
- **目标检测**：识别画面中的各种对象（人、车、物等）
- **姿态估计**：识别人体关键点，如头部、手臂、腿部位置
- **FPS显示**：实时显示处理帧率

#### 相机2（右侧）
- **目标检测**：与相机1相同
- **SAM模型**：对检测到的对象进行精确分割，显示不同颜色的掩码
- **推理时间**：显示模型处理每帧的时间

## 六、故障排除

### 1. 常见问题及解决方案

#### 问题1：相机无法打开
- **原因**：USB相机连接问题或驱动问题
- **解决方案**：
  - 检查USB线缆连接
  - 尝试更换USB端口
  - 重启系统后重新尝试

#### 问题2：Web界面刷新或卡顿
- **原因**：浏览器并发请求处理问题
- **解决方案**：系统已启用多线程模式，无需额外设置

#### 问题3：SAM模型推理失败
- **原因**：模型文件缺失或参数错误
- **解决方案**：
  - 确认 `yolo26n-seg.engine` 文件存在
  - 检查模型路径配置是否正确

#### 问题4：USB带宽不足
- **原因**：两个相机同时工作时带宽占用过高
- **解决方案**：系统已启用MJPEG压缩，无需额外设置

### 2. 日志查看

系统启动后，您可以通过以下方式查看详细日志：

```bash
# 查看容器日志
docker logs dual-camera-system

# 实时查看日志
docker logs -f dual-camera-system
```

### 3. 系统重启

如果系统出现问题，您可以通过以下步骤重启：

```bash
# 停止并删除现有容器
docker rm -f dual-camera-system

# 重新启动系统
./run_dual_camera_final_fixed.sh
```

## 七、高级配置

### 1. 调整相机参数

如果需要调整相机参数，可以修改启动脚本中的相关配置：

- **分辨率**：`self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)`
- **帧率**：`self.cap.set(cv2.CAP_PROP_FPS, fps)`
- **压缩格式**：`fourcc = cv2.VideoWriter_fourcc(*'MJPG')`

### 2. 更换模型

如果需要使用不同的YOLO模型，只需将新的 `.engine` 文件放入 `/home/seeed/ultralytics_data` 目录，并更新启动脚本中的模型路径。

### 3. 扩展功能

系统支持以下扩展：
- 添加更多相机（最多支持4个）
- 集成其他视觉模型（如OCR、行为识别等）
- 添加AI分析结果的存储和分析功能

## 八、性能优化

### 1. 推理速度优化

- 使用更小的模型（如 `yolo26n` 而非 `yolo26s`）
- 降低相机分辨率
- 减少检测对象类别

### 2. 系统稳定性优化

- 使用高质量的USB线缆
- 确保电源供应充足
- 避免同时运行其他占用GPU资源的程序

## 九、总结

本教程介绍了如何从零开始搭建一个基于YOLOv26模型和TensorRT加速的双USB相机图像处理系统。该系统具有以下特点：

- **高性能**：TensorRT加速，推理速度快
- **多功能**：目标检测、姿态估计、图像分割
- **易使用**：Web界面预览，操作简单
- **稳定性**：MJPEG压缩，降低USB带宽占用
- **可扩展**：支持添加更多相机和功能

通过本教程的指导，您应该能够成功搭建并运行这个强大的视觉处理系统，为您的项目或应用提供实时的视觉分析能力。

---

**注意**：本系统针对NVIDIA Jetson设备优化，在其他平台上可能需要适当调整配置。

**更新时间**：2026年1月21日