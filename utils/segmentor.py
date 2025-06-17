# segmentor.py
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
import cv2
from utils.visualize import draw_box_and_mask, draw_box, draw_box_tracks
from ultralytics.engine.results import Boxes

def process_frame(model, frame, transform_matrix, max_width, max_height, colors, track_history, track_time_history):
    frame_corrected = cv2.warpPerspective(frame, transform_matrix, (int(max_width), int(max_height)))
    results = model.track(frame_corrected, verbose=False, persist=True) 
    result = results[0]
    masks = result.masks
    boxes = result.boxes
    names = result.names
    
    # 指定要保留的類別名稱: person=0, suitcase=28 
    allowed_classes = {0, 28}

    if masks is not None and boxes is not None and boxes.shape[0] > 0:
        filtered_indices = []

        for i in range(boxes.shape[0]):
            cls_id = int(boxes.cls[i])

            if cls_id in allowed_classes:
                filtered_indices.append(i)
        # breakpoint()
        # 過濾 mask 與 box
        filtered_masks = masks[filtered_indices]
        filtered_boxes = {
            # 'xyxy': boxes.xyxy[filtered_indices],
            'conf': boxes.conf[filtered_indices],
            'cls': boxes.cls[filtered_indices],
            'id': boxes.id[filtered_indices] if hasattr(boxes, 'id') else None,
            'data': boxes.data[filtered_indices],
            # 'xywh': boxes.xywh[filtered_indices],
            # 'xywhn': boxes.xywhn[filtered_indices],
            # 'xyxyn': boxes.xyxyn[filtered_indices],
        }
        img = result.orig_img.copy()
        for i in range(filtered_boxes['conf'].shape[0]):
            x1, y1, x2, y2 = map(int, filtered_boxes['data'][i, :4])
            cls_id = int(filtered_boxes['cls'][i].item())
            track_id = int(filtered_boxes['id'][i].item()) if filtered_boxes['id'] is not None else -1
            label = f'{names[cls_id]} ID:{track_id}' if track_id >= 0 else f'{names[cls_id]}'
            color = colors.get(cls_id, (0, 255, 0))
            # mask = filtered_masks.data[i]
            # img = draw_box_and_mask(img, (x1, y1, x2, y2), mask, label, color)
            img = draw_box_tracks(img, (x1, y1, x2, y2), label, color, track_id, track_history, track_time_history)
        return img

    else:
        img = frame_corrected