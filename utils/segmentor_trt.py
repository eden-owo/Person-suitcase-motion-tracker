# segmentor_trt.py
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
import cv2
import numpy as np
from utils.visualize import draw_box_and_mask, draw_box, draw_box_tracks

def process_frame(model, frame, transform_matrix, max_width, max_height, colors,
                  track_history, track_time_history, track_box_history):
    import numpy as np
    import cv2

    # 做投影矯正
    frame_corrected = cv2.warpPerspective(frame, transform_matrix, (int(max_width), int(max_height)))

    # TensorRT 推論
    results = model.predict(frame_corrected, device=0)

    if not results or len(results) == 0:
        return frame_corrected.copy()

    img = frame_corrected.copy()
    result = results[0]  # 處理第一張圖片結果

    # 處理物件框
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # (x1, y1, x2, y2)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"cls:{cls_id} {conf:.2f}"
            color = colors.get(cls_id, (0, 255, 0))
            track_id = -1  # 沒有追蹤器可用

            img = draw_box_tracks(
                img, xyxy, label, color,
                track_id, track_history, track_time_history, track_box_history
            )

    return img


def process_face(model, frame):
    results = model(frame, verbose=False)
    if not results or results[0] is None:
        print("No results from model()")
        return frame.copy()

    result = results[0]

    if not (hasattr(result, 'boxes') and result.boxes and
            hasattr(result.boxes, 'data') and
            hasattr(result, 'names') and result.names):
        print("Missing boxes or names")
        return frame.copy()

    boxes, names = result.boxes, result.names

    if boxes.shape[0] == 0 or boxes.cls is None:
        return frame.copy()

    img = frame.copy()

    for i in range(boxes.shape[0]):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        # 限制在畫面範圍內，避免越界
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # 擷取臉部區域並進行模糊
        face_region = img[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)

        # 替換原本臉部區域
        img[y1:y2, x1:x2] = blurred_face

        # （可選）加框與標籤
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img, f"{names[int(boxes.cls[i])]} {boxes.conf[i]:.2f}",
        #             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img
