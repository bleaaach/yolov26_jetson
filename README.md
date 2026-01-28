# yolov26_jetson
# YOLOv26 Dual USB Camera Image Processing System Tutorial

## System Introduction

This tutorial will guide you through building a dual USB camera image processing system based on YOLOv26 model and TensorRT acceleration from scratch. The system includes the following features:

- **Dual Camera Parallel Processing**: Real-time video stream processing for two USB cameras simultaneously
- **Multi-task Vision Analysis**: Object detection, pose estimation, image segmentation (SAM model)
- **TensorRT Acceleration**: Significantly improve inference speed using NVIDIA TensorRT engine
- **Web Interface Preview**: View processing results in real-time through a browser
- **MJPEG Compression**: Reduce USB bandwidth usage and improve system stability

## Table of Contents

1. [Download from GitHub](#download-from-github)
2. [Docker Environment Setup](#docker-environment-setup)
3. [Local Environment Setup](#local-environment-setup)
4. [Model Download and Preparation](#model-download-and-preparation)
5. [System Configuration](#system-configuration)
6. [Using the System](#using-the-system)
7. [Troubleshooting](#troubleshooting)

---

## Download from GitHub

### Method 1: Clone the Entire Repository (Recommended)

This is the simplest method to get all files at once:

```bash
# 1. Navigate to your home directory
cd /home/seeed

# 2. Clone the repository
git clone https://github.com/bleaaach/yolov26_jetson.git

# 3. Navigate to the project directory
cd yolov26_jetson

# 4. View downloaded files
ls -la
```

You should see the following files:
- `run_dual_camera_docker.sh` - Docker deployment script
- `run_dual_camera_local.sh` - Local deployment script
- `README.md` - This documentation file

### Method 2: Download Individual Files

If you only need specific script files, you can download them individually using wget:

```bash
# 1. Navigate to your home directory
cd /home/seeed

# 2. Create project directory
mkdir -p yolov26_jetson
cd yolov26_jetson

# 3. Download Docker script
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/run_dual_camera_docker.sh

# 4. Download Local script
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/run_dual_camera_local.sh

# 5. Download README documentation
wget https://raw.githubusercontent.com/bleaaach/yolov26_jetson/main/README.md

# 6. View downloaded files
ls -la
```

### Verify Downloaded Files

After downloading, verify that the files exist:

```bash
# View file list
ls -la

# Check file permissions
```

If the scripts don't have execute permissions, add them:

```bash
# Add execute permissions
chmod +x run_dual_camera_docker.sh
chmod +x run_dual_camera_local.sh

# Check again
ls -la
```

### Comparison of Two Deployment Methods

| Feature | Docker Method | Local Method |
|---------|---------------|--------------|
| **Environment Isolation** | ✅ Fully isolated, doesn't pollute host environment | ❌ Directly installed on host |
| **Deployment Speed** | ✅ Fast, one-click startup | ❌ Requires manual installation of many dependencies |
| **Hardware Access** | ⚠️ Requires device mapping configuration | ✅ Direct access to all hardware |
| **Performance** | ⚠️ Has container overhead | ✅ Better performance |
| **Storage Space** | ⚠️ Requires ~2GB Docker image | ✅ Less storage usage |
| **Recommended Use Case** | Quick testing, multi-device deployment | Production environment, best performance |

---

## Docker Environment Setup

### Docker Setup Steps

#### Step 1: Ensure Docker is Installed

First check if Docker is installed:

```bash
# Check Docker version
docker --version
```

If Docker is not installed, install it first:

```bash
# Update package list
sudo apt update

# Install Docker
sudo apt install docker.io -y

# Start Docker service
sudo systemctl start docker

# Add current user to docker group
sudo usermod -aG docker $USER

# Re-login to apply changes
newgrp docker
```

#### Step 2: Ensure Docker Service is Running

```bash
# Check Docker service status
sudo systemctl status docker
```

If Docker is not running, start it:

```bash
# Start Docker service
sudo systemctl start docker
```

#### Step 3: Prepare Model Files

Ensure model files are downloaded to the correct location:

```bash
# Check model directory
ls -la /home/seeed/ultralytics_data/

# Create directory if it doesn't exist
mkdir -p /home/seeed/ultralytics_data
```

If model files don't exist, download them first:

```bash
# Navigate to model directory
cd /home/seeed/ultralytics_data

# Download object detection model
yolo export model=yolov26n.pt format=engine device=0

# Download pose estimation model
yolo export model=yolov26n-pose.pt format=engine device=0

# Download segmentation model
yolo export model=yolov26n-seg.pt format=engine device=0

# Verify model files
ls -la
```

You should see the following files:
- `yolo26n.engine`
- `yolo26n-pose.engine`
- `yolo26n-seg.engine`

#### Step 4: Run Docker Script

Now you can run the Docker script:

```bash
# 1. Navigate to project directory
cd /home/seeed/yolov26_jetson

# 2. Ensure script has execute permissions
chmod +x run_dual_camera_docker.sh

# 3. Run Docker script
./run_dual_camera_docker.sh
```

#### Step 5: Observe Startup Process

After running the script, you will see the following output:

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

#### Step 6: Wait for Docker Image Download (First Run)

If this is the first run, Docker will automatically download the Ultralytics image:

```
Unable to find image 'ultralytics/ultralytics:latest-jetson-jetpack6' locally
Pulling from library/ultralytics/ultralytics:latest-jetson-jetpack6
...
```

This process may take a few minutes depending on network speed.

#### Step 7: Wait for System Startup

After the image download completes, the system will start automatically:

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

#### Step 8: Access Web Interface

Open the following address in your browser:

```
http://localhost:5000
```

You will see real-time video streams from both cameras.

### Docker Container Management

#### View Running Containers

```bash
# View all running containers
docker ps
```

Output example:
```
CONTAINER ID   IMAGE                                      COMMAND                  STATUS    PORTS     NAMES
abc123def456   ultralytics/ultralytics:latest-jetson-jetpack6   "bash -c 'pip..."   Up 2 hours  5000/tcp   dual-camera-system
```

#### View Container Logs

```bash
# View all logs
docker logs dual-camera-system

# View logs in real-time (press Ctrl+C to exit)
docker logs -f dual-camera-system
```

#### Stop Container

```bash
# Stop container
docker stop dual-camera-system
```

#### Remove Container

```bash
# Remove container
docker rm dual-camera-system
```

#### Force Stop and Remove Container

```bash
# Force stop and remove container
docker rm -f dual-camera-system
```

#### Restart System

If you need to restart the system:

```bash
# 1. Stop and remove existing container
docker rm -f dual-camera-system

# 2. Re-run script
cd /home/seeed/yolov26_jetson
./run_dual_camera_docker.sh
```

### Docker Script Working Principle

The Docker script performs the following operations:

1. **Clean existing containers: Remove previous dual-camera-system container
2. **Check model files: Verify all necessary model files exist
3. **Create Python script: Generate Python file containing complete system code
4. **Start Docker container: Start container with the following configuration:
   - `--privileged`: Grant container privileges to allow USB device access
   - `--runtime=nvidia`: Use NVIDIA container runtime
   - `--gpus all`: Enable all available GPUs
   - `--ipc=host`: Use host IPC namespace for better performance
   - `--network=host`: Use host network for easy web server access
   - `--device=/dev/video0`: Map USB camera device to container
   - `--device=/dev/video1`: Map USB camera device to container
   - `--device=/dev/video2`: Map USB camera device to container
   - `--device=/dev/video3`: Map USB camera device to container
   - `-v "$MODEL_DIR:/models"`: Map model file directory to container
   - `-v "$(pwd)/dual_camera_system.py:/app/dual_camera_system.py"`: Map Python script to container
5. **Detect available cameras: Automatically identify USB cameras in the system
6. **Initialize cameras: Enable MJPEG compression, set resolution and frame rate
7. **Load models: Load YOLOv26 models into memory
8. **Start web server: Start Flask server in container

---

## Local Environment Setup

### Local Setup Steps

#### Step 1: Update Package List and Install pip

```bash
# Update package list
sudo apt update

# Install pip
sudo apt install python3-pip -y

# Upgrade pip
pip install -U pip
```

If pip is not pre-installed on the system, use the following command to install:

```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip
python3 get-pip.py --user
```

#### Step 2: Install Ultralytics Package

Install Ultralytics and its optional dependencies (for model export):

```bash
# Install Ultralytics
~/.local/bin/pip install ultralytics[export]
```

#### Step 3: Install PyTorch and Torchvision

**Important**: PyTorch and Torchvision installed via pip are not compatible with Jetson's ARM64 architecture. You need to manually install versions built specifically for Jetson.

First uninstall incompatible versions:

```bash
# Uninstall incompatible versions
~/.local/bin/pip uninstall torch torchvision -y
```

Then install JetPack 6.1 compatible versions:

```bash
# Install PyTorch 2.5.0
~/.local/bin/pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Install Torchvision 0.20
~/.local/bin/pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

If GitHub download is slow, you can use an acceleration proxy:

```bash
# Use gh proxy to download PyTorch
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl

# Use gh proxy to download Torchvision
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

#### Step 4: Install cuSPARSELt

cuSPARSELt is a dependency of PyTorch 2.5.0 and needs to be installed separately:

```bash
# Install cuSPARSELt
sudo apt-get install -y libcusparselt0
```

#### Step 5: Install onnxruntime-gpu

onnxruntime-gpu is used for some model export functions. Since the package on PyPI doesn't contain aarch64 binaries for Jetson, manual installation is required:

```bash
# Install onnxruntime-gpu 1.23.0
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

Or use version 1.20.0:

```bash
# Install onnxruntime-gpu 1.20.0
~/.local/bin/pip install https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

#### Step 6: Configure PATH Environment Variable

Since user installation mode is used, pip installed executables are located in the `~/.local/bin/` directory. It's recommended to add this directory to the PATH environment variable:

```bash
# Add to .bashrc
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc

# Reload .bashrc
source ~/.bashrc
```

#### Step 7: Verify Installation

Check installed package versions:

```bash
# Check versions
python3 -c "import ultralytics; import torch; import torchvision; import onnxruntime; print('ultralytics version:', ultralytics.__version__); print('torch version:', torch.__version__); print('torchvision version:', torchvision.__version__); print('onnxruntime version:', onnxruntime.__version__)"
```

Expected output:

```
ultralytics version: 8.4.7
torch version: 2.5.0a0+872d972e41.nv24.08
torchvision version: 0.20.0a0+afc54f7
onnxruntime version: 1.23.0
```

#### Step 8: Test YOLOv26 Inference Functionality

```python
from ultralytics import YOLO
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Load YOLOv26n model
model = YOLO('yolo26n.pt')
print(f"Model loaded successfully!")

# Perform inference test
results = model('https://ultralytics.com/images/bus.jpg')
print(f"Inference successful! Detected {len(results[0].boxes)} objects")

# Display detection results
for i, box in enumerate(results[0].boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = model.names[cls_id]
    print(f"  Object {i+1}: {cls_name} (confidence: {conf:.2f})")
```

#### Step 9: Prepare Model Files

Ensure model files are downloaded to the correct location:

```bash
# Check model directory
ls -la /home/seeed/ultralytics_data/

# Create directory if it doesn't exist
mkdir -p /home/seeed/ultralytics_data
```

If model files don't exist, download them first:

```bash
# Navigate to model directory
cd /home/seeed/ultralytics_data

# Download object detection model
yolo export model=yolov26n.pt format=engine device=0

# Download pose estimation model
yolo export model=yolov26n-pose.pt format=engine device=0

# Download segmentation model
yolo export model=yolov26n-seg.pt format=engine device=0

# Verify model files
ls -la
```

You should see the following files:
- `yolo26n.engine`
- `yolo26n-pose.engine`
- `yolo26n-seg.engine`

#### Step 10: Run Local Script

Now you can run the Local script:

```bash
# 1. Navigate to project directory
cd /home/seeed/yolov26_jetson

# 2. Ensure script has execute permissions
chmod +x run_dual_camera_local.sh

# 3. Run Local script
./run_dual_camera_local.sh
```

#### Step 11: Observe Startup Process

After running the script, you will see the following output:

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

#### Step 12: Wait for System Startup

The system will start automatically:

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

#### Step 13: Observe Performance Output

While the system is running, you will see performance statistics:

```
Camera 1 - Total time: 13.1ms, Model time: 12.3ms, Secondary task: New, FPS: 30.0
Camera 2 - Total time: 25.4ms, Model time: 24.5ms, Secondary task: New, FPS: 9.9
```

### Local Script Working Principle

The Local script performs the following operations:

1. **Check display environment: Detect if a display environment is available
2. **Create virtual display: If no display environment, try to create virtual display with Xvfb
3. **Disable OpenCV GUI: If virtual display cannot be created, disable OpenCV GUI features
4. **Detect available cameras: Automatically identify USB cameras in the system
5. **Initialize cameras: Start cameras using V4L2 mode with Zero-Copy DMA enabled
6. **Load models: Load YOLOv26 models into memory
7. **Start camera processors: Start independent processing threads for each camera
8. **Main loop: Continuously process camera frames until user presses 'q' to exit

---

## Model Download and Preparation

### Model File Directory

Model files are stored in the `/home/seeed/ultralytics_data` directory. If this directory doesn't exist, create it first:

```bash
# Create model directory
mkdir -p /home/seeed/ultralytics_data

# Navigate to model directory
cd /home/seeed/ultralytics_data
```

### Download YOLOv26 Models

Use the following commands to download YOLOv26 series models:

```bash
# Download object detection model
yolo export model=yolov26n.pt format=engine device=0

# Download pose estimation model
yolo export model=yolov26n-pose.pt format=engine device=0

# Download segmentation model (SAM)
yolo export model=yolov26n-seg.pt format=engine device=0
```

**Parameter Description**:
- `model`: Model file to export
- `format=engine`: Export as TensorRT engine format
- `device=0`: Use GPU 0 for export

### Verify Model Files

After downloading, confirm model files exist:

```bash
# View model files
ls -la /home/seeed/ultralytics_data/
```

You should see the following files:
- `yolo26n.engine` (object detection)
- `yolo26n-pose.engine` (pose estimation)
- `yolo26n-seg.engine` (image segmentation)

### Model File Size Reference

| Model | Size (approx) | Purpose |
|-------|---------------|---------|
| yolo26n.engine | 12 MB | Object detection |
| yolo26n-pose.engine | 12 MB | Pose estimation |
| yolo26n-seg.engine | 12 MB | Image segmentation |

---

## System Configuration

### Check USB Cameras

Connect two USB cameras, then check camera devices:

```bash
# View all video devices
ls -la /dev/video*
```

The system should recognize two or more video devices (such as `/dev/video0`, `/dev/video1`, etc.).

Output example:
```
crw-rw----+ 1 root video 81, 0 Dec 31  1969 /dev/video0
crw-rw----+ 1 root video 81, 1 Dec 31  1969 /dev/video1
crw-rw----+ 1 root video 81, 2 Dec 31  1969 /dev/video2
crw-rw----+ 1 root video 81, 3 Dec 31  1969 /dev/video3
```

### Startup Script Configuration

The startup scripts are configured with the following parameters:

| Parameter | Docker Method | Local Method |
|-----------|---------------|--------------|
| **Camera Resolution** | 320x240 / 640x360 | 640x480 |
| **Frame Rate** | 30fps / 25fps | 30fps |
| **Compression Format** | MJPEG | Zero-Copy DMA |
| **Web Server** | http://localhost:5000 | OpenCV window |
| **Multi-threading** | Enabled | Enabled |
| **Display Method** | Web browser | OpenCV window |

---

## Using the System

### Docker Method: Access Web Interface

Open in your browser:

```
http://localhost:5000
```

You will see real-time video streams from both cameras, including:
- Camera 1: Object detection + Pose estimation
- Camera 2: Object detection + SAM model segmentation

### Local Method: View Output

The system will display results in an OpenCV window. If no display environment is available, it will run in headless mode and only process frames without display.

Press 'q' to exit the system.

---

## Troubleshooting

### Issue 1: Camera Not Detected

**Symptom**: System shows "No cameras found" or cannot access camera devices.

**Solution**:
```bash
# Check if camera devices exist
ls -la /dev/video*

# Check camera permissions
sudo chmod 666 /dev/video0
sudo chmod 666 /dev/video2

# Add user to video group
sudo usermod -aG video $USER
```

### Issue 2: Model File Not Found

**Symptom**: System shows "Model file not found" error.

**Solution**:
```bash
# Check model directory
ls -la /home/seeed/ultralytics_data/

# If directory doesn't exist, create it
mkdir -p /home/seeed/ultralytics_data

# Download models
cd /home/seeed/ultralytics_data
yolo export model=yolov26n.pt format=engine device=0
yolo export model=yolov26n-pose.pt format=engine device=0
yolo export model=yolov26n-seg.pt format=engine device=0
```

### Issue 3: Docker Container Cannot Start

**Symptom**: Docker container fails to start or exits immediately.

**Solution**:
```bash
# Check Docker logs
docker logs dual-camera-system

# Check if Docker service is running
sudo systemctl status docker

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Remove and recreate container
docker rm -f dual-camera-system
cd /home/seeed/yolov26_jetson
./run_dual_camera_docker.sh
```

### Issue 4: Low Frame Rate

**Symptom**: System frame rate is lower than expected.

**Solution**:
```bash
# Enable maximum performance mode
sudo nvpmodel -m 0

# Enable maximum clock frequency
sudo jetson_clocks

# Monitor system performance
tegrastats

# Check GPU utilization
nvidia-smi
```

### Issue 5: Out of Memory Error

**Symptom**: System shows CUDA out of memory error.

**Solution**:
- Reduce camera resolution
- Reduce model size (use yolov26n instead of yolov26s)
- Close other GPU applications
- Increase swap space

### Issue 6: ImportError or ModuleNotFoundError

**Symptom**: System shows import errors when running.

**Solution**:
```bash
# Reinstall dependencies
~/.local/bin/pip install --upgrade ultralytics[export]

# Verify installation
python3 -c "import ultralytics; print(ultralytics.__version__)"

# Check PyTorch installation
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Advanced Configuration

### Performance Optimization

To achieve maximum performance, run the following commands before starting the system:

```bash
# Set to maximum performance mode
sudo nvpmodel -m 0

# Enable maximum clock frequency
sudo jetson_clocks

# Disable power management features
sudo jetson_clocks --show
```

### Custom Model Configuration

You can modify the model configuration in the script:

```python
# Change model path
DETECTION_MODEL = "/path/to/your/model.engine"

# Change camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Change frame rate
CAMERA_FPS = 30

# Change inference interval
POSE_ESTIMATION_INTERVAL = 2  # Run pose estimation every 2 frames
SEGMENTATION_INTERVAL = 5     # Run segmentation every 5 frames
```

### Adding More Cameras

To add more cameras, modify the camera configuration:

```python
# Add more camera indices
AVAILABLE_CAMERAS = [0, 2, 4, 6]  # Add more camera indices
```

---

## Summary

This tutorial provides a comprehensive guide for setting up and running a dual USB camera image processing system based on YOLOv26 and TensorRT on NVIDIA Jetson devices. The system features:

- **High Performance**: Achieves 30fps real-time processing through Zero-Copy optimization
- **Flexible Deployment**: Supports both Docker and local deployment methods
- **Multi-task Processing**: Simultaneously performs object detection, pose estimation, and segmentation
- **Easy to Use**: Simple one-click startup with comprehensive error handling
- **Scalable**: Supports adding more cameras and features

With the guidance of this tutorial, you should be able to successfully build and run this powerful vision processing system, providing real-time vision analysis capabilities for your projects or applications.

---

**Note**: This system is optimized for NVIDIA Jetson devices and may require appropriate configuration adjustments on other platforms.

**Last Updated**: January 28, 2026
