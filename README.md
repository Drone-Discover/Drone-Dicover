
```markdown
# Drone Discover 🚁🔍

**Drone Discover** is an AI-powered computer vision project designed to detect visual pollution and identify structural/building non-compliances from aerial and drone imagery using **YOLOv5**.

---

## 📌 Project Overview

Urban planning and municipal compliance require fast, automated, and accurate monitoring. **Drone Discover** processes high-resolution aerial footage to detect key visual violations across cities:

* **Visual Pollution Detection:** Identifies litter, waste accumulation, graffiti, and clutter using target classes (`garbage`, `cans`, `trash`, `glass`, `metals`, `grafitti`, `water`).
* **Building Violation Monitoring:** Detects unauthorized construction, encroaching structures, and non-compliant building modifications.

---

## 📁 Repository Structure

```text
Drone-Discover/
│
├── Code/                           # Training scripts, inference utilities, and custom helpers
├── Data/                           # Dataset configuration (coco.yaml with dataset paths & class labels)
├── Dataset/                        # Annotated aerial image datasets (YOLO format)
├── Presentation & Report & Poster/ # Project documentation, poster, and presentation slides
└── runs/train/exp5/                # Training output runs (YOLO weights, evaluation metrics, and result images)

```

---

## 🎯 Target Classes (`Data/coco.yaml`)

The model is configured to detect and label the following visual pollution classes:

* `0`: Garbage
* `1`: Cans
* `2`: Trash
* `3`: Glass
* `4`: Metals
* `5`: Grafitti
* `6`: Water

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python
* **Framework:** YOLOv5 (Ultralytics), PyTorch
* **Computer Vision & Utilities:** OpenCV, Matplotlib, NumPy, Pandas, PyYAML

---

## 🚀 Getting Started

1. **Clone the repository:**
```bash
git clone https://github.com/EngEsraa1/Drone-Discover.git
cd Drone-Discover

```


2. **Install requirements:**
```bash
pip install -r requirements.txt

```


3. **Train Model:**
```bash
python Code/train.py --img 640 --batch 16 --epochs 50 --data Data/coco.yaml --weights yolov5s.pt

```


4. **Run Inference:**
```bash
python Code/detect.py --weights runs/train/exp5/weights/best.pt --source Dataset/

```



---

## 📊 Evaluation & Output

All trained model weights (`best.pt`, `last.pt`), performance curves ($F1$, $Precision$, $Recall$, Confusion Matrix), and sample prediction grids are organized inside the `runs/train/exp5/` directory.

```

```
