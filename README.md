# 🔫 Automated Weapon Detection & Evidence Filtering System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered forensic tool that automatically detects weapons in crime scene images using YOLOv8 deep learning — reducing manual triage time by up to 99.5%**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📌 Problem Statement

Forensic investigators manually review **500–1000 crime scene photos per case**, spending **8–12 hours** on initial evidence triage. This process is:
- ❌ Time-consuming and prone to human fatigue
- ❌ Inconsistent across different investigators  
- ❌ A bottleneck that delays critical investigations

**AWDEFS solves this** by automating weapon detection and filtering relevant evidence in **under 5 minutes**.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🎯 **Real-time Detection** | Detects firearms and knives using YOLOv8-nano |
| 🖼️ **Annotated Output** | Bounding boxes with confidence scores overlaid |
| 📁 **Batch Filtering** | Process entire folders of crime scene images |
| 📊 **CSV Reports** | Auto-generated forensic documentation |
| 🖥️ **Web Interface** | Clean Streamlit UI for non-technical users |
| ⚡ **CPU Compatible** | No GPU required — runs on standard forensic workstations |

---

## 🎬 Demo

### Web Application
Upload any image and get instant weapon detection results:

```
streamlit run app.py
```

### Command Line Detection
```bash
python detect.py --source input/ --output output/
```

### Evidence Filtering
```bash
python run_filter.py --classes weapon_gun,weapon_knife --annotate
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- 4GB RAM minimum
- No GPU required

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anshuldhanjwal/Automated-Weapon-Detection-System.git
cd Automated-Weapon-Detection-System
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

---

## 🛠️ Usage

### 🌐 Web App (Recommended for beginners)
```bash
streamlit run app.py
# Open http://localhost:8501 in your browser
# Upload any image → See detections instantly
```

### 💻 Command Line Detection
```bash
# Detect on a single image
python detect.py --source test.jpg

# Detect on a folder of images
python detect.py --source input/
```

### 📂 Evidence Filtering Pipeline
```bash
# Filter images containing guns only
python run_filter.py --classes weapon_gun --annotate

# Filter images containing knives only
python run_filter.py --classes weapon_knife

# Filter all weapon types
python run_filter.py --classes weapon_gun,weapon_knife --annotate
```

**Output generated:**
```
output/
├── filtered_images/       # Images containing detected weapons
├── annotated_images/      # Same images with bounding boxes drawn
├── detection_report.csv   # Per-detection metadata
└── summary_report.txt     # Statistics summary
```

---

## 🏗️ Architecture

```
Input Image (JPG/PNG)
        │
        ▼
┌──────────────────┐
│  Preprocessor    │  Resize → 416×416, Normalize pixels
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│   YOLOv8-nano Model          │
│  ├── Backbone (CSPDarknet)   │  Feature extraction
│  ├── Neck (FPN)              │  Multi-scale fusion
│  └── Head (Detection)       │  Bounding box + class prediction
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────┐
│  Post-Processing │  NMS → Filter by confidence threshold
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│  Output Generator           │
│  ├── Annotated images       │
│  ├── CSV detection report   │
│  └── Summary statistics     │
└─────────────────────────────┘
```

---

## 📊 Results

### Performance on Test Dataset (11 images)

| Metric | Value |
|--------|-------|
| Total images scanned | 11 |
| Images with weapons detected | 7 (63.6%) |
| Firearm detections | 5 |
| Knife detections | 4 |
| Average confidence score | 0.86 |
| Processing speed | ~120ms per image (CPU) |
| Total processing time | 2.3 seconds |

### Time Savings vs Manual Process

| Process | Manual | AWDEFS | Improvement |
|---------|--------|--------|-------------|
| Initial triage | 480 min | 2.3 min | **99.5% faster** |
| Report generation | 120 min | <1 min | **99.2% faster** |
| Total | 600 min | 3.3 min | **99.5% faster** |

### Training Convergence

| Epoch | Loss | mAP@50 |
|-------|------|--------|
| 1/30 | 1.494 | 0.453 |
| 15/30 | 0.652 | 0.629 |
| 30/30 | 0.375 | **0.771** |

---

## 📁 Project Structure

```
Automated-Weapon-Detection-System/
│
├── app.py                  # Streamlit web application
├── detect.py               # Core detection script
├── run_filter.py           # Evidence filtering pipeline
├── requirements.txt        # Python dependencies
├── README.md               # You are here
│
├── input/                  # Place your crime scene images here
│   └── sample_scene.jpg
│
├── output/                 # Auto-generated results
│   ├── filtered_images/
│   ├── annotated_images/
│   ├── detection_report.csv
│   └── summary_report.txt
│
└── .devcontainer/          # GitHub Codespaces config
    └── devcontainer.json
```

---

## 🔬 Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Core language | 3.10+ |
| YOLOv8 (Ultralytics) | Object detection model | 8.0+ |
| OpenCV | Image processing & annotation | 4.5+ |
| NumPy | Numerical operations | 1.21+ |
| Pillow | Image I/O | 8.0+ |
| Streamlit | Web UI | Latest |
| PyTorch | Deep learning backend | 2.0+ (CPU) |

---

## ⚠️ Limitations

- Trained on handguns and knives only (no rifles, explosives, etc.)
- Performance may degrade in extreme low-light images
- Designed for static image analysis (video processing is a future enhancement)
- Investigators must verify all AI-flagged evidence before legal use

---

## 🔭 Future Scope

- [ ] Real-time video stream processing
- [ ] GUI application (PyQt5)
- [ ] Bloodstain and trace evidence detection
- [ ] LIMS (Laboratory Information Management System) integration
- [ ] Cloud deployment (AWS/Azure)
- [ ] Multi-dataset training for higher accuracy

---

## 📄 Academic Context

> This project was developed as part of the B.Tech–M.Tech (CSE - Cyber Security) program at **National Forensic Sciences University, Delhi Campus**, under the supervision of **Dr. Brahm Prakash**, Department of Cyber Security & Digital Forensics.

---

## 📚 References

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- Weapon-Kaggle Dataset: https://www.kaggle.com/datasets/snehilsanyal/weapon-detection-test
- Redmon & Farhadi (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made by <a href="https://github.com/Anshuldhanjwal">Anshul Dhanjwal</a> | NFSU Delhi
</div>
