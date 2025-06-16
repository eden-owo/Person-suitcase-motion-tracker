# segmentor.py
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
import cv2
from utils.visualize import draw_box_and_mask

def process_frame(model, frame, transform_matrix, max_width, max_height, colors):
    frame_corrected = cv2.warpPerspective(frame, transform_matrix, (int(max_width), int(max_height)))
    results = model(frame_corrected) 
    result = results[0]
    masks = result.masks
    boxes = result.boxes
    names = result.names
    
    # 指定要保留的類別名稱
    allowed_classes = {"person", "suitcase"}

    if masks is not None and boxes is not None and boxes.shape[0] > 0:
        filtered_indices = []

        for i in range(boxes.shape[0]):
            cls_id = int(boxes.data[i, 5].item())
            cls_name = names[cls_id]
            if cls_name in allowed_classes:
                filtered_indices.append(i)

        # 過濾 mask 與 box
        if filtered_indices:
            result.masks.data = result.masks.data[filtered_indices]
            result.boxes.data = result.boxes.data[filtered_indices]
            if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                result.boxes.id = result.boxes.id[filtered_indices]

            img = result.orig_img.copy()
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = map(int, boxes.data[i, :4])
                cls_id = int(boxes.data[i, 5].item())
                label =  f'{names[cls_id]}'
                mask = masks.data[i]
                color = colors.get(cls_id, (0, 255, 0))
                img = draw_box_and_mask(img, (x1, y1, x2, y2), mask, label, color)
            return img
        else:
            img = frame_corrected  # 沒有目標就回傳矯正後的影像
    else:
        img = frame_corrected

