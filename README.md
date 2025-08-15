# Instance Segmentation of People and Suitcases Using YOLOv11 and Custom C++ OpenCV Build


This repository showcases instance segmentation of people and suitcases using YOLOv11 models and a self-compiled OpenCV C++ library.

With calculating the velocity of people and suitcases, its primary goal is to detect any instances of falls involving either.

<figure><img src="https://github.com/eden-owo/Person-suitcase-motion-tracker/blob/master/pics/demo.png" alt=""><figcaption></figcaption></figure>

TensorRT is recommended for at least 30% faster computation.

| GPU | Model | With TensorRT (INT8) | Without TensorRT |
|:--:|:--:|:--:|:--:|
| GTX 1650 Laptop | Yolo11m-seg | 22 FPS | 16 FPS |
| RTX 4060 Laptop | Yolo11m-seg | 48 FPS | 32 FPS |
| Jetson orin nano super | Yolo11m-seg | 24 FPS | 12 FPS |

Flask is used for web-based visualization, as OpenCV’s GTK interface is not supported in headless environments like Docker or Kubernetes.

<figure><img src="https://github.com/eden-owo/Person-suitcase-motion-tracker/blob/master/pics/suitcase-alarm.png" alt=""><figcaption></figcaption></figure>

🧱 Architecture


The project structure is as follows, with a description of each module's functionality and purpose:


```text
project/
│
├── main.py                # Main entry point of the program, responsible for argument parsing and overall flow control
├── photo_PR_example.py    # Example script for performing geometric transformations on a single image
│
├── utils/                 # Utility modules
│   ├── segmentor.py       # Performs real-time instance segmentation using a preloaded YOLO model after applying perspective correction
│   ├── transform.py       # Provides ROI selection and perspective transformation functions
│   ├── video_utils.py     # Handles video reading, saving, and GPU-accelerated processing
│   └── visualize.py       # Renders bounding boxes and segmentation masks
│
└── yolo/
    └── yolo_seg.py        # Encapsulation and inference logic for the YOLOv8 Segmentation model

```
