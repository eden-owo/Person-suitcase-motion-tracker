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

Flask is applied for web visualization, rather than opencv-gtk in command-based case.

<figure><img src="https://github.com/eden-owo/Person-suitcase-motion-tracker/blob/master/pics/suitcase-alarm.png" alt=""><figcaption></figcaption></figure>

🧱 Architecture


專案結構如下，說明各模組功能與用途：


```text
project/
│
├── main.py                # 主程式入口點，負責參數解析與整體流程控制
├── photo_PR_example.py    # 單張影像進行幾何變形的範例程式
│
├── utils/                 # 實用工具模組
│   ├── segmentor.py       # 使用已載入的 YOLO 模型對輸入影像進行透視矯正後的即時實例分割（Instance Segmentation）。
│   ├── transform.py       # 提供 ROI 選取與透視變換（perspective transform）功能
│   ├── video_utils.py     # 影片讀取、儲存與 GPU 加速處理
│   └── visualize.py       # 負責繪製 bounding boxes 與 segmentation masks
│
└── yolo/
    └── yolo_seg.py        # YOLOv8 Segmentation 模型的封裝與推論邏輯

```
