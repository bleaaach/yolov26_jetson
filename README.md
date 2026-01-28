# yolov26_jetson
# YOLOv26 双 USB 相机图像处理系统教程

## 系统介绍

本教程将从零开始，教您如何搭建一个基于 YOLOv26 模型和 TensorRT 加速的双 USB 相机图像处理系统。该系统具有以下功能：

- **双相机并行处理**：同时支持两个 USB 相机的实时视频流处理
- **多任务视觉分析**：目标检测、姿态估计、图像分割（SAM 模型）
- **TensorRT 加速**：使用 NVIDIA TensorRT 引擎大幅提升推理速度
- **Web 界面预览**：通过浏览器实时查看处理结果
- **MJPEG 压缩**：降低 USB 带宽占用，提高系统稳定性

## 目录

1. [从 GitHub 下载](#从-github-下载)
2. [Docker 环境运行](#docker-环境运行)
3. [本地环境运行](#本地环境运行)
4. [模型下载与准备](#模型下载与准备)
5. [系统配置](#系统配置)
6. [使用系统](#使用系统)
7. [故障排除](#故障排除)

---

## 从 GitHub 下载

### 方法一：克隆整个仓库（推荐）

这是最简单的方法，可以一次性获取所有文件：

```bash
# 1. 导航到您的家目录
cd /home/seeed

# 2. 克隆仓库
git clone https://github.com/bleaaach/yolov26_jetson.git

# 3. 进入项目目录
cd yolov26_jetson

# 4. 查看下载的文件
ls -la
```

您应该看到以下文件：
- `run_dual_camera_docker.sh` - Docker 部署脚本
- `run_dual_camera_local.sh` - 本地部署脚本
- `README.md` - 本文档文件

### 方法二：单独下载文件

如果您只需要特定的脚本文件，可以使用 wget 命令单独下载：

```bash
# 1. 导航到您的家目录
cd /home/seeed

# 2. 创建项目目录
mkdir -p yolov26_jetson
cd yolov26_jetson

# 3. 下载 Docker 脚本
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/run_dual_camera_docker.sh

# 4. 下载 Local 脚本
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/run_dual_camera_local.sh

# 5. 下载 README 文档
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/README.md

# 6. 查看下载的文件
ls -la
```

### 验证下载的文件

下载完成后，验证文件是否存在：

```bash
# 查看文件列表
ls -la

# 检查文件权限
是否可执行
```

如果脚本没有执行权限，需要添加：

```bash
# 添加执行权限
chmod +x run_dual_camera_docker.sh
chmod +x run_dual_camera_local.sh

# 再次检查
ls -la
```

### 两种部署方式的区别

| 特性 | Docker 方式 | Local 方式 |
|------|-------------|-------------|
| **环境隔离** | ✅ 完全隔离，不污染主机环境 | ❌ 直接安装在主机上 |
| **部署速度** | ✅ 快速，一键启动 | ❌ 需要手动安装依赖很多依赖 |
| **硬件访问** | ⚠️ 需要配置设备映射 | ✅ 直接访问所有硬件 |
| **性能** | ⚠️ 有容器开销 | ✅ 更好的性能 |
| **存储空间** | ⚠️ 需要约 2GB 的 Docker 镜像 | ✅ 更少的存储占用 |
| **推荐场景** | 快速测试、多设备部署 | 生产环境、追求最佳性能 |

---

## Docker 环境运行

### Docker 方式运行步骤

#### 步骤 1：确保 Docker 已安装

首先检查 Docker 是否已安装：

```bash
# 检查 Docker 版本
docker --version
```

如果 Docker 未安装，请先安装 Docker：

```bash
# 更新软件包列表
sudo apt update

# 安装 Docker
sudo apt install docker.io -y

# 启动 Docker 服务
sudo systemctl start docker

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录以使更改生效
newgrp docker
```

#### 步骤 2：确保 Docker 服务正在运行

```bash
# 检查 Docker 服务状态
sudo systemctl status docker
```

如果 Docker 未运行，启动它：

```bash
# 启动 Docker 服务
sudo systemctl start docker
```

#### 步骤 3：准备模型文件

确保模型文件已下载到正确位置：

```bash
# 检查模型目录
ls -la /home/seeed/ultralytics_data/

# 如果目录不存在，创建它
mkdir -p /home/seeed/ultralytics_data
```

如果模型文件不存在，需要先下载：

```bash
# 进入模型目录
cd /home/seeed/ultralytics_data

# 下载目标检测模型
yolo export model=yolov26n.pt format=engine device=0

# 下载姿态估计模型
yolo export model=yolov26n-pose.pt format=engine device=0

# 下载分割模型
yolo export model=yolov26n-seg.pt format=engine device=0

# 验证模型文件
ls -la
```

您应该看到以下文件：
- `yolo26n.engine`
- `yolo26n-pose.engine`
- `yolo26n-seg.engine`

#### 步骤 4：运行 Docker 脚本

现在可以运行 Docker 脚本了：

```bash
# 1. 导航到项目目录
cd /home/seeed/yolov26_jetson

# 2. 确保脚本有执行权限
chmod +x run_dual_camera_docker.sh

# 3. 运行 Docker 脚本
./run_dual_camera_docker.sh
```

#### 步骤 5：观察启动过程

脚本运行后，您将看到以下输出：

```
=== Jetson Performance Optimization Suggestions ===
Run the following commands for best performance:
  sudo nvpmodel -m 0  # Maximum performance mode
  sudo jetson_clocks  # Enable maximum clock frequency
  tegrastats  # Monitor system performance

=== TensorRT FP16/INT8 Mixed Quantization Optimization Tips ===
Check if TensorRT engines use mixed quantization:
  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n.engine --fp16
  trtexec --loadEngine=/home/seeed/ultralytics_data/yolo26n-pose.engine --int8

=== Dual USB Camera Image Processing System Startup Script ===
Based on YOLOv26 model and TensorRT acceleration

System configuration:
- Camera 1: Object detection + Pose estimation
- Camera 2: Object detection + SAM model
- Web server: http://localhost:5000

Model directory: /home/seeed/ultralytics_data

Press Ctrl+C to exit

=== Checking Model Files ===
✅ Model file check completed
```

#### 步骤 6：等待 Docker 镜像下载（首次运行）

如果是第一次运行，Docker 会自动下载 Ultralytics 镜像：

```
Unable to find image 'ultralytics/ultralytics:latest-jetson-jetpack6' locally
Pulling from library/ultralytics/ultralytics:latest-jetson-jetpack6
...
```

这个过程可能需要几分钟，取决于网络速度。

#### 步骤 7：等待系统启动

镜像下载完成后，系统将自动启动：

```
=== Finding available cameras ===
✅ Camera index 0 available
✅ Camera index 2 available
Found 2 available cameras: [0, 2]

✅ Detection model loaded - FP16: False, INT8: False
✅ Pose model loaded - FP16: False, INT8: False
✅ Segmentation model loaded - FP16: False, INT8: False

✅ Camera 0 (Camera 1) opened successfully (V4L2)
✅ Camera 0 test read successful
✅ Camera 0 (Camera 1) processing thread started
✅ Camera 1 started successfully

✅ Camera 2 (Camera 2) opened successfully (V4L2)
✅ Camera 2 test read successful
✅ Camera 2 (Camera 2) processing thread started
✅ Camera 2 started successfully

✅ Web server started successfully
Please visit in browser: http://localhost:5000

=== Dual camera system startup completed ===
Available cameras: Camera1 Camera2
Press Ctrl+C to exit system
```

#### 步骤 8：访问 Web 界面

在浏览器中打开以下地址：

```
http://localhost:5000
```

您将看到两个相机的实时视频流。

### Docker 容器管理

#### 查看运行中的容器

```bash
# 查看所有运行中的容器
docker ps
```

输出示例：
```
CONTAINER ID   IMAGE                                      COMMAND                  STATUS    PORTS     NAMES
abc123def456   ultralytics/ultralytics:latest-jetson-jetpack6   "bash -c 'pip..."   Up 2 hours  5000/tcp   dual-camera-system
```

#### 查看容器日志

```bash
# 查看所有日志
docker logs dual-camera-system

# 实时查看日志（按 Ctrl+C 退出）
docker logs -f dual-camera-system
```

#### 停止容器

```bash
# 停止容器
docker stop dual-camera-system
```

#### 删除容器

```bash
# 删除容器
docker rm dual-camera-system
```

#### 强制停止并删除容器

```bash
# 强制停止并删除容器
docker rm -f dual-camera-system
```

#### 重启系统

如果需要重启系统：

```bash
# 1. 停止并删除现有容器
docker rm -f dual-camera-system

# 2. 重新运行脚本
cd /home/seeed/yolov26_jetson
./run_dual_camera_docker.sh
```

### Docker 脚本工作原理

Docker 脚本会执行以下操作：

1. **清理现有容器**：移除之前的 dual-camera-system 容器
2. **检查模型文件**：验证所有必要的模型文件是否存在
3. **创建 Python 脚本**：生成包含完整系统代码的 Python 文件
4. **启动 Docker 容器**：使用以下配置启动容器：
   - `--privileged`：授予容器特权，允许访问 USB 设备
   - `--runtime=nvidia`：使用 NVIDIA 容器运行时
   - `--gpus all`：启用所有可用的 GPU
   - `--ipc=host`：使用主机的 IPC 命名空间，提高性能
   - `--network=host`：使用主机网络，便于 Web 服务器访问
   - `--device=/dev/video0`：映射 USB 相机设备到容器
   - `--device=/dev/video1`：映射 USB 相机设备到容器
   - `--device=/dev/video2`：映射 USB 相机设备到容器
   - `--device=/dev/video3`：映射 USB 相机设备到容器
   - `-v "$MODEL_DIR:/models"`：映射模型文件目录到容器
   - `-v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py"`：映射 Python 脚本到容器
5. **检测可用相机**：自动识别系统中的 USB 相机
6. **初始化相机**：启用 MJPEG 压缩，设置分辨率和帧率
7. **加载模型**：加载 YOLOv26 模型到内存
8. **启动 Web 服务器**：在容器中启动 Flask 服务器

---

## 本地环境运行

### Local 方式运行步骤

#### 步骤 1：更新软件包列表并安装 pip

```bash
# 更新软件包列表
sudo apt update

# 安装 pip
sudo apt install python3-pip -y

# 升级 pip
pip install -U pip
```

如果系统没有预装 pip，可以使用以下命令安装：

```bash
# 下载 get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# 安装 pip
python3 get-pip.py --user
```

#### 步骤 2：安装 Ultralytics 包

安装 Ultralytics 及其可选依赖项（用于模型导出）：

```bash
# 安装 Ultralytics
~/.local/bin/pip install ultralytics[export]
```

#### 步骤 3：安装 PyTorch 和 Torchvision

**重要**：通过 pip 安装的 PyTorch 和 Torchvision 与 Jetson 的 ARM64 架构不兼容，需要手动安装专门为 Jetson 构建的版本。

首先卸载不兼容的版本：

```bash
# 卸载不兼容的版本
~/.local/bin/pip uninstall torch torchvision -y
```

然后安装 JetPack 6.1 兼容的版本：

```bash
# 安装 PyTorch 2.5.0
~/.local/bin/pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# 安装 Torchvision 0.20
~/.local/bin/pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

如果 GitHub 下载速度较慢，可以使用加速代理：

```bash
# 使用 gh 下载 PyTorch
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# 使用 gh 下载 Torchvision
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

#### 步骤 4：安装 cuSPARSELt

cuSPARSELt 是 PyTorch 2.5.0 的依赖项，需要单独安装：

```bash
# 安装 cuSPARSELt
sudo apt-get install -y libcusparselt0
```

#### 步骤 5：安装 onnxruntime-gpu

onnxruntime-gpu 用于某些模型导出功能。由于 PyPI 上的包不包含 aarch64 Jetson 的二进制文件，需要手动安装：

```bash
# 安装 onnxruntime-gpu 1.23.0
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

或者使用 1.20.0 版本：

```bash
# 安装 onnxruntime-gpu 1.20.0
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

#### 步骤 6：配置 PATH 环境变量

由于使用了用户安装模式，pip 安装的可执行文件位于 `~/.local/bin/` 目录。建议将该目录添加到 PATH 环境变量中：

```bash
# 添加到 .bashrc
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc

# 重新加载 .bashrc
source ~/.bashrc
```

#### 步骤 7：验证安装

检查已安装的包版本：

```bash
# 检查版本
python3 -c "import ultralytics; import torch; import torchvision; import onnxruntime; print('ultralytics version:', ultralytics.__version__); print('torch version:', torch.__version__); print('torchvision version:', torchvision.__version__); print('onnxruntime version:', onnxruntime.__version__)"
```

预期输出：

```
ultralytics version: 8.4.7
torch version: 2.5.0a0+872d972e41.nv24.08
torchvision version: 0.20.0a0+afc54f7
onnxruntime version: 1.23.0
```

#### 步骤 8：测试 YOLOv26 推理功能

```python
from ultralytics import YOLO
import torch

# 检查 CUDA 是否可用
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    print(f"CUDA 设备名称: {torch.cuda.get_device_name(0)}")

# 加载 YOLOv26n 模型
model = YOLO('yolo26n.pt')
print(f"模型加载成功！")

# 进行推理推理测试
results = model('https://ultralytics.com/images/bus.jpg')
print(f"推理成功！检测到 {len(results[0].boxes)} 个目标")

# 显示检测结果
for i, box in enumerate(results[0].boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = model.names[cls_id]
    print(f"  目标 {i+1}: {cls_name} (置信度: {conf:.2f})")
```

#### 步骤 9：准备模型文件

确保模型文件已下载到正确位置：

```bash
# 检查模型目录
ls -la /home/seeed/ultralytics_data/

# 如果目录不存在，创建它
mkdir -p /home/seeed/ultralytics_data
```

如果模型文件不存在，需要先下载：

```bash
# 进入模型目录
cd /home/seeed/ultralytics_data

# 下载目标检测模型
yolo export model=yolov26n.pt format=engine device=0

# 下载姿态估计模型
yolo export model=yolov26n-pose.pt format=engine device=0

# 下载分割模型
yolo export model=yolov26n-seg.pt format=engine device=0

# 验证模型文件
ls -la
```

您应该看到以下文件：
- `yolo26n.engine`
- `yolo26n-pose.engine`
- `yolo26n-seg.engine`

#### 步骤 10：运行 Local 脚本

现在可以运行 Local 脚本了：

```bash
# 1. 导航到项目目录
cd /home/seeed/yolov26_jetson

# 2. 确保脚本有执行权限
chmod +x run_dual_camera_local.sh

# 3. 运行 Local 脚本
./run_dual_camera_local.sh
```

#### 步骤 11：观察启动过程

脚本运行后，您将看到以下输出：

```
=== Jetson Performance Optimization Suggestions ===
Run the following commands for best performance:
  sudo nvpmodel -m 0  # Maximum performance mode
  sudo jetson_clocks  # Enable maximum clock frequency
  tegrastats  # Monitor system performance

=== Dual USB Camera Image Processing System Startup Script (Local Run - Zero-Copy Optimized) ===
Based on YOLOv26 model and TensorRT acceleration

System configuration:
- Camera 1: Object detection (every frame) + Pose estimation (every 2 frames)
- Camera 2: Object detection (every frame) + Segmentation (every 5 frames)
- Display method: OpenCV window
- Camera resolution: 640x480 @ 30fps
- Inference precision: FP16 (half precision)
- Optimization: Zero-Copy DMA + Async inference + Shared model

Model directory: /home/seeed/ultralytics_data

Press 'q' to exit

=== Checking Model Files ===
✅ Model file check completed

=== Starting local Python script ===
⚠️ No display environment detected, trying to create virtual display with Xvfb...
⚠️ Xvfb not installed, trying to disable OpenCV GUI...
⚠️ No display environment detected, skipping window display (frame processing only)
```

#### 步骤 12：等待系统启动

系统将自动启动：

```
=== Finding available cameras ===
✅ Camera index 0 available
✅ Camera index 2 available
Found 2 available cameras: [0, 2]

✅ Detection model loaded: /home/seeed/ultralytics_data/yolo26n.engine
✅ Pose model loaded: /home/seeed/ultralytics_data/yolo26n-pose.engine
✅ Segmentation model loaded: /home/seeed/ultralytics_data/yolo26n-seg.engine

=== Starting Zero-Copy camera capture 0 (Camera 1) ===
✅ Camera 0 V4L2 mode started successfully
✅ Camera processor started: Camera 1
✅ Camera 1 started successfully

=== Starting Zero-Copy camera capture 2 (Camera 2) ===
✅ Camera 2 V4L2 mode started successfully
✅ Camera processor started: Camera 2
✅ Camera 2 started successfully

=== Dual camera system startup completed ===
Available cameras: Camera 1 Camera 2
Press 'q' to exit system

=== Headless mode running - processing frames only, no display ===
Press Ctrl+C to exit
```

#### 步骤 13：观察性能输出

系统运行时，您将看到性能统计信息：

```
Camera 1 - Total time: 13.1ms, Model time: 12.3ms, Secondary task: New, FPS: 30.0
Camera 2 - Total time: 25.4ms, Model time: 24.5ms, Secondary task: New, FPS: 9.9
```

### Local 脚本工作原理

Local 脚本会执行以下操作：

1. **检查显示环境**：检测是否有可用的显示环境
2. **创建虚拟显示**：如果没有显示环境，尝试使用 Xvfb 创建虚拟显示
3. **禁用 OpenCV GUI**：如果无法创建虚拟显示，禁用 OpenCV 的 GUI 功能
4. **检测可用相机**：自动识别系统中的 USB 相机
5. **初始化相机**：使用 V4L2 模式启动相机，启用 Zero-Copy DMA
6. **加载模型**：加载 YOLOv26 模型到内存
7. **启动相机处理器**：为每个相机启动独立的处理线程
8. **主循环**：持续处理相机帧，直到用户按 'q' 键退出

---

## 模型下载与准备

### 模型文件目录

模型文件存储在 `/home/seeed/ultralytics_data` 目录中。如果该目录不存在，请先创建：

```bash
# 创建模型目录
mkdir -p /home/seeed/ultralytics_data

# 进入模型目录
cd /home/seeed/ultralytics_data
```

### 下载 YOLOv26 模型

使用以下命令下载 YOLOv26 系列模型：

```bash
# 下载目标检测模型
yolo export model=yolov26n.pt format=engine device=0

# 下载姿态估计模型
yolo export model=yolov26n-pose.pt format=engine device=0

# 下载分割模型（SAM）
yolo export model=yolov26n-seg.pt format=engine device=0
```

**参数说明**：
- `model`：要导出的模型文件
- `format=engine`：导出为 TensorRT 引擎格式
- `device=0`：使用 GPU 0 进行导出

### 验证模型文件

下载完成后，确认模型文件存在：

```bash
# 查看模型文件
ls -la /home/seeed/ultralytics_data/
```

您应该看到以下文件：
- `yolo26n.engine`（目标检测）
- `yolo26n-pose.engine`（姿态估计）
- `yolo26n-seg.engine`（图像分割）

### 模型文件大小参考

| 模型 | 大小（约） | 用途 |
|------|-------------|------|
| yolo26n.engine | 12 MB | 目标检测 |
| yolo26n-pose.engine | 12 MB | 姿态估计 |
| yolo26n-seg.engine | 12 MB | 图像分割 |

---

## 系统配置

### 检查 USB 相机

连接两个 USB 相机，然后检查相机设备：

```bash
# 查看所有视频设备
ls -la /dev/video*
```

系统应该识别到两个或更多的视频设备（如 `/dev/video0`、`/dev/video1` 等）。

输出示例：
```
crw-rw----+ 1 root video 81, 0 Dec 31  1969 /dev/video0
crw-rw----+ 1 root video 81, 1 Dec 31  1969 /dev/video1
crw-rw----+ 1 root video 81, 2 Dec 31  1969 /dev/video2
crw-rw----+ 1 root video 81, 3 Dec 31  1969 /dev/video3
```

### 启动脚本配置

启动脚本已配置好以下参数：

| 参数 | Docker 方式 | Local 方式 |
|------|-------------|-------------|
| **相机分辨率** | 320x240 / 640x360 | 640x480 |
| **帧率** | 30fps / 25fps | 30fps |
| **压缩格式** | MJPEG | Zero-Copy DMA |
| **Web 服务器** | http://localhost:5000 | OpenCV 窗口 |
| **多线程** | 已启用 | 已启用 |
| **显示方式** | Web 浏览器 | OpenCV 窗口 |

---

## 使用系统

### Docker 方式访问 Web 界面

在浏览器中打开：

```
http://localhost:5000
```

您将看到两个相机的实时视频流，包括：
- 相机 1：目标检测 + 姿态估计
- 相机 2：目标检测 + SAM 模型分割

### Local 方式查看输出

Local 方式在终端中显示性能信息：

```
Camera 1 - Total time: 13.1ms, Model time: 12.3ms, Secondary task: New, FPS: 30.0
Camera 2 - Total time: 25.4ms, Model time: 24.5ms, Secondary task: New, FPS: 9.9
```

按 'q' 键退出系统。

### 系统功能说明

#### 相机 1（左侧）
- **目标检测**：识别画面中的各种对象（人、车、物等）
- **姿态估计**：识别人体关键点，如头部、手臂、腿部位置
- **FPS 显示**：实时显示处理帧率

#### 相机 2（右侧）
- **目标检测**：与相机 1 相同
- **SAM 模型**：对检测到的对象进行精确分割，显示不同颜色的掩码
- **推理时间**：显示模型处理每帧的时间

---

## 故障排除

### 常见问题及解决方案

#### 问题 1：相机无法打开

**原因**：USB 相机连接问题或驱动问题

**解决方案**：
- 检查 USB 线缆连接
- 尝试更换 USB 端口
- 重启系统后重新尝试

**调试命令**：
```bash
# 查看视频设备
ls -la /dev/video*

# 检查 USB 设备
lsusb

# 查看内核日志
dmesg | grep -i usb
```

#### 问题 2：Web 界面刷新或卡顿

**原因**：浏览器并发请求处理问题

**解决方案**：系统已启用多线程模式，无需额外设置

#### 问题 3：SAM 模型推理失败

**原因**：模型文件缺失或参数错误

**解决方案**：
- 确认 `yolo26n-seg.engine` 文件存在
- 检查模型路径配置是否正确

**调试命令**：
```bash
# 检查模型文件
ls -la /home/seeed/ultralytics_data/

# 检查模型文件大小
du -h /home/seeed/ultralytics_data/*.engine
```

#### 问题 4：USB 带宽不足

**原因**：两个相机同时工作时带宽占用过高

**解决方案**：系统已启用 MJPEG 压缩，无需额外设置

#### 问题 5：Docker 容器无法启动

**原因**：Docker 服务未运行或权限问题

**解决方案**：
```bash
# 检查 Docker 服务状态
sudo systemctl status docker

# 启动 Docker 服务
sudo systemctl start docker

# 检查用户是否在 docker 组中
groups

# 将用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录
newgrp docker
```

#### 问题 6：模型加载失败

**原因**：模型文件损坏或版本不兼容

**解决方案**：
```bash
# 删除损坏的模型文件
rm /home/seeed/ultralytics_data/*.engine

# 重新下载模型
cd /home/seeed/ultralytics_data
yolo export model=yolov26n.pt format=engine device=0
yolo export model=yolov26n-pose.pt format=engine device=0
yolo export model=yolov26n-seg.pt format=engine device=0
```

### 日志查看

#### Docker 方式

```bash
# 查看容器日志
docker logs dual-camera-system

# 实时查看日志（按 Ctrl+C 退出）
docker logs -f dual-camera-system

# 查看最后 100 行日志
docker logs --tail 100 dual-camera-system
```

#### Local 方式

Local 方式的输出直接显示在终端中，无需额外命令。

### 系统重启

#### Docker 方式

```bash
# 停止并删除现有容器
docker rm -f dual-camera-system

# 重新运行脚本
cd /home/seeed/yolov26_jetson
./run_dual_camera_docker.sh
```

#### Local 方式

```bash
# 停止脚本（按 Ctrl+C）
# 然后重新运行
cd /home/seeed/yolov26_jetson
./run_dual_camera_local.sh
```

---

## 高级配置

### 调整相机参数

如果需要调整相机参数，可以修改启动脚本中的相关配置：

- **分辨率**：`self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)`
- **帧率**：`self.cap.set(cv2.CAP_PROP_FPS, fps)`
- **压缩格式**：`fourcc = cv2.VideoWriter_fourcc(*'MJPG')`

### 更换模型

如果需要使用不同的 YOLO 模型，只需将新的 `.engine` 文件放入 `/home/seeed/ultralytics_data` 目录，并更新启动脚本中的模型路径。

### 扩展功能

系统支持以下扩展：
- 添加更多相机（最多支持 4 个）
- 集成其他视觉模型（如 OCR、行为识别等）
- 添加 AI 分析结果的存储和分析功能

---

## 性能优化

### 推理速度优化

- 使用更小的模型（如 `yolo26n` 而非 `yolo26s`）
- 降低相机分辨率
- 减少检测对象类别

### 系统稳定性优化

- 使用高质量的 USB 线缆
- 确保电源供应充足
- 避免同时运行其他占用 GPU 资源的程序

### Jetson 性能优化建议

```bash
# 设置为最大性能模式
sudo nvpmodel -m 0

# 启用最大时钟频率
sudo jetson_clocks

# 监控系统性能
tegrastats
```

---

## 总结

本教程介绍了如何从零开始搭建一个基于 YOLOv26 模型和 TensorRT 加速的双 USB 相机图像处理系统。该系统具有以下特点：

- **高性能**：TensorRT 加速，推理速度快
- **多功能**：目标检测、姿态估计、图像分割
- **易使用**：Web 界面预览，操作简单
- **稳定性**：MJPEG 压缩，降低 USB 带宽占用
- **可扩展**：支持添加更多相机和功能

通过本教程的指导，您应该能够成功搭建并运行这个强大的视觉处理系统，为您的项目或应用提供实时的视觉分析能力。

---

**注意**：本系统针对 NVIDIA Jetson 设备优化，在其他平台上可能需要适当调整配置。

**最后更新时间**：2026 年 1 月 28 日
