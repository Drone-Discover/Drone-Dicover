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
