---
description: This guide shows how to build a dual GMSL camera image processing system based on YOLOv26 model and TensorRT acceleration on Jetson platforms. It specifically targets the Sensing SG3S-ISX031C-GMSL2F cameras on reComputer J4012/AGX Orin, achieving 1920x1536@30fps performance.
title: YOLOv26 Dual GMSL Camera Image Processing System on Jetson
keywords:
  - reComputer
  - YOLOv26
  - Dual GMSL Camera
  - TensorRT
  - SG3S-ISX031C-GMSL2F
  - Computer Vision
  - Jetson
  - Object Detection
  - Pose Estimation
  - Image Segmentation
image: https://files.seeedstudio.com/wiki/yolov26_on_jetson/local.png
slug: /ai_robotics_yolov26_dual_gmsl_system
sku: 100090853,100076722,100060802,100032662
last_update:
  date: 02/10/2026
  author: AI Assistant
---

<div style={{ textAlign: "justify" }}>
This wiki demonstrates how to build a high-performance dual GMSL camera image processing system based on YOLOv26 model and TensorRT acceleration. Designed for industrial applications, this system leverages the GMSL interface for high-bandwidth, low-latency video transmission.
</div>

## User Scenario

> **"Customers will use reComputer J4012 (or AGX Orin) to build a dual SG3S-ISX031C-GMSL2F camera real-time visual inspection system, utilizing TensorRT acceleration to achieve high-frame-rate detection at 1920x1536@30fps, to evaluate Jetson's performance and stability in industrial multi-camera scenarios."**

## Key Features

- **High-Resolution GMSL Capture**: Supports dual [Sensing SG3S-ISX031C-GMSL2F](https://www.seeedstudio.com/SG3S-ISX031C-GMSL2F-p-6245.html) cameras at **1920x1536** resolution.
- **Guaranteed 30FPS Performance**: Optimized pipeline ensures stable **30 FPS** capture and processing using DMA zero-copy and hardware acceleration.
- **TensorRT-Only Inference**: Removes PyTorch runtime overhead by using pure TensorRT engines for maximum throughput.
- **Multi-task Vision Analysis**: Concurrent Object Detection, Pose Estimation, and Instance Segmentation.
- **Industrial Optimization**:
    - **Input**: 1920x1536 @ 30fps (Raw YUY2)
    - **Process**: 640x480 (Hardware Resized)
    - **Latency**: Minimized via DMA buffers and multi-threaded pipeline

## YOLOv26 Model Overview

This system utilizes the **YOLOv26** architecture, which represents the state-of-the-art in real-time computer vision. Specifically, we deploy the **Nano (n)** series models, which are tailored for edge AI devices like the NVIDIA Jetson Orin.

### Why YOLOv26 on Jetson?

According to the [Ultralytics Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/), YOLOv26 offers several distinct advantages for embedded deployment:

1.  **High Efficiency on ARM64**: The model architecture is optimized for the ARM64 processor architecture found in Jetson devices, ensuring low power consumption while maintaining high throughput.
2.  **Tensor Core Acceleration**: When exported to **TensorRT**, YOLOv26 fully leverages the dedicated Tensor Cores in the Jetson Orin's Ampere GPU architecture. This allows for:
    -   **Low Latency**: Critical for real-time industrial inspection.
    -   **High Throughput**: Capable of processing multiple high-resolution streams simultaneously.
3.  **Unified Framework**: A single architecture supports multiple tasks (Detection, Segmentation, Pose), simplifying the deployment pipeline on limited-resource edge devices.

### Deployed Models

We run three specialized model variants concurrently:

1.  **Object Detection (`yolov26n`)**:
    -   **Task**: Bounding box detection and classification.
    -   **Classes**: 80 standard COCO classes (Person, Vehicle, etc.).
    -   **Advantage**: Extremely fast inference for primary object localization.

2.  **Pose Estimation (`yolov26n-pose`)**:
    -   **Task**: Human skeletal keypoint detection (17 keypoints).
    -   **Advantage**: Real-time behavior analysis without the need for heavy external pose libraries.

3.  **Instance Segmentation (`yolov26n-seg`)**:
    -   **Task**: Pixel-level object masking.
    -   **Advantage**: Provides precise object contours, essential for defect detection where bounding boxes are insufficient.

**Optimization Strategy**: All models are exported to **TensorRT Engine format (.engine)** with **FP16 precision**. This bypasses the PyTorch runtime overhead and maximizes the usage of the Orin's 100+ TOPS (Tera Operations Per Second) AI performance.

## Prerequisites

### Hardware
- **[reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-w-o-power-adapter-p-5628.html)** (Orin NX 16GB) or **Jetson AGX Orin**
- **2x [Sensing SG3S-ISX031C-GMSL2F Camera](https://www.seeedstudio.com/SG3S-ISX031C-GMSL2F-p-6245.html)**
- **GMSL Deserializer Board**: Compatible with Jetson Orin (ensure drivers are installed)
- **Cables**: High-quality Fakra cables

### Software
- **JetPack 6.x** (L4T 36.x)
- **GStreamer** with NVIDIA acceleration plugins (`nvv4l2camerasrc` or standard `v4l2src` with `io-mode=dmabuf`)
- **Python 3.10+**
- **Ultralytics YOLOv26**

---

## Installation & Setup

### 1. Clone the Repository

```bash
cd /home/seeed
git clone https://github.com/bleaaach/yolov26_jetson.git
cd yolov26_jetson
```

### 2. Detailed Installation Steps

Follow these steps to set up the environment completely from scratch.

#### Step 1. Update Package List and Install pip

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

#### Step 2. Install Ultralytics Package

Install Ultralytics and its optional dependencies (for model export):

```bash
# Install Ultralytics
~/.local/bin/pip install ultralytics[export]
```

#### Step 3. Install PyTorch and Torchvision

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

#### Step 4. Install cuSPARSELt

cuSPARSELt is a dependency of PyTorch 2.5.0 and needs to be installed separately:

```bash
# Install cuSPARSELt
sudo apt-get install -y libcusparselt0
```

#### Step 5. Install onnxruntime-gpu

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

#### Step 6. Configure PATH Environment Variable

Since user installation mode is used, pip installed executables are located in the `~/.local/bin/` directory. It's recommended to add this directory to the PATH environment variable:

```bash
# Add to .bashrc
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc

# Reload .bashrc
source ~/.bashrc
```

#### Step 7. Verify Installation

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

#### Step 8. Test YOLOv26 Inference Functionality

```python
from ultralytics import YOLO
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Load YOLOv26n model
model = YOLO('yolov26n.pt')
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

#### Step 9. Prepare Model Files

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
yolo export model=yolov26n.pt format=engine device=0 half=True

# Download pose estimation model
yolo export model=yolov26n-pose.pt format=engine device=0 half=True

# Download segmentation model
yolo export model=yolov26n-seg.pt format=engine device=0 half=True

# Verify model files
ls -la
```

You should see the following files:
- `yolo26n.engine`
- `yolo26n-pose.engine`
- `yolo26n-seg.engine`

#### Step 10. Run Local Script

Now you can run the Local script:

```bash
# 1. Navigate to project directory
cd /home/seeed/yolov26_jetson

# 2. Ensure script has execute permissions
chmod +x run_dual_gmsl_local.sh

# 3. Run Local script
./run_dual_gmsl_local.sh
```

---

## Configuration & Running

The system uses `run_dual_gmsl_local.sh` which is pre-configured for the SG3S-ISX031C camera.

### Camera Configuration Details

The script is hardcoded with the following optimal parameters for the SG3S-ISX031C camera:

| Parameter | Value | Note |
|-----------|-------|------|
| **Camera Model** | SG3S-ISX031C-GMSL2F | Sensing GMSL2 Camera |
| **Capture Resolution** | **1920 x 1536** | Full Sensor Resolution |
| **Capture FPS** | **30 FPS** | Native Frame Rate |
| **Process Resolution** | 640 x 480 | Downscaled via Hardware (VIC) for Inference |
| **Pixel Format** | YUY2 | Converted to BGR via Hardware |

### Running the System

Execute the GMSL-optimized script:

```bash
cd /home/seeed/yolov26_jetson
chmod +x run_dual_gmsl_local.sh
./run_dual_gmsl_local.sh
```

### Expected Output

1. **Console**: You will see "AGX Orin GMSL Camera Ultimate 30fps Edition" and hardware status (CPU/GPU frequencies).
2. **Display**: A window showing dual camera feeds with:
    - Object Detection boxes (thickened for visibility)
    - Pose Estimation skeletons (Full 17-point COCO format)
    - Instance Segmentation masks
    - Real-time FPS counter (Target: 30fps)

---

## Performance Notes

To achieve the stable 30fps target on high-resolution streams, the system employs:
1. **DMA Zero-Copy**: Uses `io-mode=dmabuf` in GStreamer to pass image data directly from camera to GPU memory without CPU copying.
2. **Hardware Conversion**: Uses `nvvidconv` for color space conversion (YUY2 -> BGRx) and resizing.
3. **Asynchronous Inference**: Detection, Pose, and Segmentation run on separate intervals to balance load.
4. **Jetson Clocks**: The script automatically attempts to maximize performance (`nvpmodel -m 0`, `jetson_clocks`).

## Troubleshooting

- **Low FPS?**: Ensure you are using the `.engine` models, not `.pt`. The script enforces `.engine` usage.
- **No Video?**: Check GMSL connections and ensure `/dev/video*` devices exist. Verify drivers with `v4l2-ctl --list-devices`.
- **Memory Issues**: The system is optimized for 8GB+ RAM. If using J4012 (16GB), performance should be optimal.

## Resources

- [Sensing SG3S-ISX031C-GMSL2F Product Page](https://www.seeedstudio.com/SG3S-ISX031C-GMSL2F-p-6245.html)
- [NVIDIA Jetson Download Center](https://developer.nvidia.com/embedded/downloads)
- [Ultralytics YOLOv26 on NVIDIA Jetson Guide](https://docs.ultralytics.com/guides/nvidia-jetson/)
