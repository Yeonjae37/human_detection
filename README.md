# Real-Time Human Detection & Tracking

This repository combines **YOLOv8** and **[SAM2](https://github.com/facebookresearch/sam2)** (with motion modeling from **[SAMURAI](https://github.com/yangchris11/samurai)**) into a single end-to-end pipeline for real-time person detection, pixel-precise segmentation, PTZ camera tracking, and anomaly alerts.

### About SAM2
**SAM2** (Segment Anything Model 2) is designed for object segmentation and tracking but lacks built-in capabilities 
for performing this in real time.

### About SAMURAI
**SAMURAI** enhances SAM2 by introducing motion modeling, leveraging temporal motion cues for better 
tracking accuracy without retraining or fine-tuning.  


## Project Structure
```
human_detection/
├── src/                   # Source code
│   ├── sam2/              # SAM2 library
│   └── human_detection/   # Main package
│       ├── __init__.py
│       ├── app.py         # Main application
│       ├── config/        # Configuration files
│       │   └── settings.py
│       ├── models/        # Model implementations
│       │   ├── detector.py
│       │   └── tracker.py
│       ├── utils/         # Utilities
│       │   ├── alerts.py
│       │   └── visualization.py
│       └── web/           # Web interface
│           ├── routes.py
│           └── templates/
│               └── index.html
├── tests/               # Test code
├── checkpoints/         # Model checkpoints
├── setup.py             # Package configuration
├── pyproject.toml       # Build system configuration
└── README.md
```

## Key Features

- **YOLOv8 Person Detection**  
  Ultra-fast bounding-box detection of humans in each video frame.

- **SAM2 Segmentation & Tracking**  
  Pixel-accurate masks + centroid extraction to hand off to the PTZ controller.

- **Motion-Aware Tracking**  
  SAMURAI motion modeling ensures stable multi-object tracks without retraining.

- **Anomaly Alerting**  
  Detects "stop-and-interact" behavior (e.g., a thief grabbing an object) and generates an alert.


---

## Setup Instructions
I recommend using uv venv to create isolated environments, simplifying dependency management and ensuring reproducible setups.

### 1. Create & activate virtualenv
```bash
# Install the 'uv' CLI and create a new venv
pip install uv
uv venv

# On macOS / Linux
source .venv/bin/activate
# On Windows (PowerShell)
source .venv/Scripts/activate
```

### 2. Clone the repository
```bash
git clone https://github.com/Yeonjae37/human_detection.git
```

### 3. Install packages
```bash
cd human_detection

# Install the core package (SAM2 + demo app) in editable mode
uv pip install -e .

# If you want to use GPU acceleration (CUDA), you must install PyTorch with the correct CUDA version manually.
# For example, to install PyTorch with CUDA 12.1, run the following command before installing the rest:
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

### 4. Download SAM2 Checkpoints
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

---

## Usage
### Web Application
Run the web application for real-time human detection and tracking:

```bash
python -m human_detection.app
```

After running, access the web interface at `http://localhost:5000`.

### Customization
You can customize various settings in `src/human_detection/config/settings.py`:

- **Camera Settings**: Change `CAMERA_INDEX` to use different cameras
- **Model Paths**: Update `YOLO_MODEL_PATH` and `SAM_CHECKPOINT_PATH` to use different model checkpoints
- **Detection Thresholds**: Adjust `STATIONARY_THRESHOLD` and `MASK_THRESHOLD` to fine-tune detection sensitivity

### Acknowledgment
This project leverages:  
- **YOLOv8** by Ultralytics for ultra-fast real-time person detection.  
- **SAM2** by Meta FAIR for pixel-precise segmentation and tracking.  
- **SAMURAI** by the University of Washington's Information Processing Lab for motion-aware memory modeling.  


## Citation
```
@article{glenn2024yolov8,
  title={YOLOv8: Next-Generation Real-Time Object Detection},
  author={Glenn Jocher and Ultralytics},
  year={2024},
  url={https://github.com/ultralytics/ultralytics}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi et al.},
  year={2024},
  url={https://arxiv.org/abs/2408.00714}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting SAM for Zero-Shot Visual Tracking with Motion-Aware Memory},
  author={Yang et al.},
  year={2024},
  url={https://arxiv.org/abs/2411.11922}
}

```

---
