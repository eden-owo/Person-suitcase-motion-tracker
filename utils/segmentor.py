# segmentor.py
from ultralytics import YOLO
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
import cv2
from utils.visualize import draw_box_and_mask

def process_frame(model, frame, transform_matrix, max_width, max_height, colors):
    frame_corrected = cv2.warpPerspective(frame, transform_matrix, (int(max_width), int(max_height)))
    results = model(frame_corrected) 
    masks = results[0].masks
    
    if masks is not None and masks.data.shape[0] > 0:   
        result = results[0] 
        img = result.orig_img.copy()
        boxes = result.boxes
        names = result.names

        if boxes is not None and boxes.shape[0] > 0:
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = map(int, boxes.data[i, :4])
                cls_id = int(boxes.data[i, 5].item())
                label =  f'{names[cls_id]}'
                mask = masks.data[i]
                color = colors.get(cls_id, (0, 255, 0))
                img = draw_box_and_mask(img, (x1, y1, x2, y2), mask, label, color)
            return img
    else:
        return frame_corrected

