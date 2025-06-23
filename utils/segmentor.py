# segmentor.py
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
import cv2
import numpy as np
from utils.visualize import draw_box_and_mask, draw_box, draw_box_tracks

def process_frame(model, frame, transform_matrix, max_width, max_height, colors, track_history, track_time_history):
    frame_corrected = cv2.warpPerspective(frame, transform_matrix, (int(max_width), int(max_height)))
    results = model.track(frame_corrected, verbose=False, persist=True)
    if not results or results[0] is None:
        print("No results from model.track()")
        return frame_corrected.copy()

    result = results[0]

    if not (hasattr(result, 'boxes') and result.boxes and
            hasattr(result.boxes, 'data') and
            hasattr(result, 'masks') and result.masks and
            hasattr(result, 'names') and result.names):
        print("Missing boxes, masks, or names")
        return frame_corrected.copy()

    boxes, masks, names = result.boxes, result.masks, result.names

    if boxes.shape[0] == 0 or boxes.cls is None:
        return frame_corrected.copy()

    allowed_classes = {0, 28}
    cls_array = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
    # 用 NumPy 過濾索引
    filtered_indices = np.where(np.isin(cls_array, list(allowed_classes)))[0]

    if filtered_indices.size == 0:
        return frame_corrected.copy()

    # 過濾 boxes 資料 (先轉 numpy，確保可索引)
    def safe_index(attr):
        if attr is None:
            return None
        arr = attr.cpu().numpy() if hasattr(attr, 'cpu') else np.array(attr)
        return arr[filtered_indices]

    filtered_conf = safe_index(boxes.conf)
    filtered_cls = safe_index(boxes.cls)
    filtered_id = safe_index(boxes.id) if hasattr(boxes, 'id') else None
    filtered_data = safe_index(boxes.data)

    if filtered_data is None or filtered_data.shape[0] == 0:
        print("Invalid or missing data tensor")
        return frame_corrected.copy()

    img = result.orig_img.copy()
    for i in range(filtered_data.shape[0]):
        try:
            x1, y1, x2, y2 = map(int, filtered_data[i, :4])
        except Exception as e:
            print(f"Failed to extract box coordinates at index {i}: {e}")
            continue

        cls_id = int(filtered_cls[i]) if filtered_cls is not None else -1
        track_id = int(filtered_id[i]) if filtered_id is not None else -1
        label = f'{names[cls_id]} ID:{track_id}' if track_id >= 0 else f'{names[cls_id]}' if cls_id >= 0 else 'Unknown'
        color = colors.get(cls_id, (0, 255, 0))
        img = draw_box_tracks(img, (x1, y1, x2, y2), label, color, track_id, track_history, track_time_history)

    return img

import cv2

def process_face(model, frame):
    results = model(frame)
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
        cls_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        label = f"{names[cls_id]} {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

        # Draw label
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img
