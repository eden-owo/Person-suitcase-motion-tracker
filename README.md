
Architecture

project/
│
├── main.py                # 主程式入口點，解析參數與流程控制
├── config.py              # 配置（如 model 路徑、class ID、顏色）
├── utils/
│   ├── video_utils.py     # 處理影片讀取、寫入、GPU 加速
│   ├── visualize.py       # 畫框與 mask 的視覺化工具
│   ├── transform.py       # ROI選取與 perspective 變換工具
│   └── timer.py           # FPS 與延遲計算
│
└── yolo/
    └── yolo_seg.py        # YOLOv8Seg 類別
