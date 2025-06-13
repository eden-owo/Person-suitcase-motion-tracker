
🧱 Architecture
專案結構如下，說明各模組功能與用途：

project/
│
├── main.py                # 主程式入口點，負責參數解析與整體流程控制
├── config.py              # 配置檔案，包含模型路徑、分類 ID、顏色定義等設定
│
├── utils/                 # 實用工具模組
│   ├── video_utils.py     # 影片讀取、儲存與 GPU 加速處理
│   ├── visualize.py       # 負責繪製 bounding boxes 與 segmentation masks
│   ├── transform.py       # 提供 ROI 選取與透視變換（perspective transform）功能
│   └── timer.py           # 計算 FPS 與處理時間的輔助工具
│
└── yolo/
    └── yolo_seg.py        # YOLOv8 Segmentation 模型的封裝與推論邏輯

