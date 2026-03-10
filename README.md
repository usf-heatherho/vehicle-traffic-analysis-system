# 🚗 Real-Time Traffic Analysis System

A real-time computer vision system that detects, tracks, and analyzes
vehicle traffic from video using modern deep learning and multi-object
tracking techniques.

This project demonstrates a full end-to-end pipeline for traffic
analytics including vehicle detection, persistent tracking, directional
counting, and live visualization.

------------------------------------------------------------------------

## 📸 Demo

![Traffic Analytics Demo](assets/demo.gif)

Example output showing: - Vehicle detection - Persistent tracking IDs -
Diagonal line-crossing analytics - Real-time traffic counts

------------------------------------------------------------------------

## 🧠 System Overview

The system processes each video frame through the following pipeline:

Video Input\
↓\
Vehicle Detection (YOLOv8)\
↓\
Multi-Object Tracking (DeepSORT)\
↓\
Line-Crossing Analytics (Geometric Detection)\
↓\
Visualization & Traffic Metrics

Vehicles are counted when their tracked centroid crosses a predefined
diagonal line, allowing direction-based traffic analytics.

------------------------------------------------------------------------

## 🚀 Features

-   Real-time vehicle detection using YOLOv8
-   Persistent multi-object tracking with DeepSORT
-   Geometric line-crossing detection using centroid-based crossing
    logic
-   Directional traffic counting (Inbound / Outbound flow)
-   Real-time visualization with bounding boxes, IDs, and traffic
    metrics
-   Modular architecture separating detection, tracking, analytics, and
    visualization

------------------------------------------------------------------------

## 🏗 Project Architecture

```
traffic-analytics-system/
│
├── assets/
|   ├── demo.gif         # Sample demo
|
├── data/
│   └── highway_traffic_15.mp4
|
├── src/
│   ├── detector.py      # YOLO vehicle detection
│   ├── tracker.py       # DeepSORT tracking
│   ├── counter.py       # Line-crossing analytics
│   └── visualizer.py    # Frame rendering & overlays
│
├── models/
|   ├── yolov8n.pt       # Detection model
│
├── run.py               # Main pipeline execution
├── requirements.txt
└── README.md
```
------------------------------------------------------------------------

## 🔧 Technologies Used

-   Python
-   OpenCV
-   YOLOv8 (Ultralytics)
-   DeepSORT Multi-Object Tracking
-   NumPy

------------------------------------------------------------------------

## 📊 Traffic Analytics Logic

Vehicle direction is determined using a diagonal line-crossing detection
algorithm.

Each tracked vehicle maintains a centroid history. When the centroid
crosses the counting line, the system detects the direction based on the
sign change of a geometric side-of-line calculation.

`prev_side * current_side < 0`

This allows the system to reliably detect vehicles moving into or out of
a traffic flow region.

------------------------------------------------------------------------

## ▶️ How to Run

### 1️. Clone the repository
```
git clone https://github.com/usf-heather-ho/traffic-analytics-system.git
cd traffic-analytics-system
```
### 2️. Install dependencies

`pip install -r requirements.txt`

### 3️. Run the application

`python run.py`

Press ESC to exit the video window.

------------------------------------------------------------------------

## 📈 Example Output

The system visualizes:

-   Vehicle bounding boxes
-   Unique tracking IDs
-   Vehicle centroids
-   Directional traffic counts
-   Diagonal counting line

------------------------------------------------------------------------

## 🎯 Future Improvements

Potential enhancements:

-   Vehicle speed estimation
-   Lane-based traffic analytics
-   Traffic density heatmaps
-   Data logging for long-term analytics
-   Web dashboard visualization

------------------------------------------------------------------------

## 📌 Why This Project

This project demonstrates practical skills in:

-   Computer Vision
-   Real-Time Systems
-   Multi-Object Tracking
-   Data Analytics from Video
-   Clean modular software design

------------------------------------------------------------------------
## 💡 Inspiration

This project was inspired by real-world work experience involving computer vision systems for monitoring vehicle traffic using camera footage.

To respect proprietary systems and internal code, this repository implements an independent version of the core concepts using publicly available tools such as YOLOv8 and DeepSORT.

The goal of this project was to recreate the key ideas behind traffic analytics systems — vehicle detection, multi-object tracking, and directional counting — in a modular and open-source environment.

-------------------------------------------------------------------------
## 👩‍💻 Author

Heather Ho\
Information Technology --- University of South Florida\
Aspiring Software Engineer

-------------------------------------------------------------------------
## 🪪 License

This project is licensed under the MIT License - see the LICENSE file for details.
