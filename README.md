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


🚀 Getting StartedClone the repository:Bashgit clone [https://github.com/EngEsraa1/Drone-Discover.git](https://github.com/EngEsraa1/Drone-Discover.git)
cd Drone-Discover
Install requirements:Bashpip install -r requirements.txt
Train Model:Bashpython Code/train.py --img 640 --batch 16 --epochs 50 --data Data/coco.yaml --weights yolov5s.pt
Run Inference:Bashpython Code/detect.py --weights runs/train/exp5/weights/best.pt --source Dataset/
📊 Evaluation & OutputAll trained model weights (best.pt, last.pt), performance curves ($F1$, $Precision$, $Recall$, Confusion Matrix), and sample prediction grids are organized inside the runs/train/exp5/ directory.
