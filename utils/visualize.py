# visualize.py

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import numpy as np
import time

def draw_box_and_mask(img, box, mask, label, color):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img 必須是 numpy.ndarray，目前是 {type(img)}")
    """
    繪製 bbox, label 和對應的 segmentation mask。

    Parameters:
        img: 原始影像（np.ndarray）
        box: bbox 座標 (x1, y1, x2, y2)
        mask: mask 二值影像（np.ndarray）
        label: 要顯示的文字
        color: RGB 顏色 (tuple)

    Returns:
        img: 處理後影像（np.ndarray）
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # mask 從 torch tensor 轉為 numpy，值為 0 或 255，並保留單通道
    mask = mask.cpu().numpy()
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # 確保 mask 尺寸與 img 相同（尤其是 H, W）
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_color = np.zeros_like(img, dtype=np.uint8)
    mask_color[:, :] = color
    
    masked = cv2.bitwise_and(mask_color, mask_color, mask=mask)
    img = cv2.addWeighted(img, 1.0, masked, 0.5, 0)

    return img

def draw_box(img, box, label, color):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img 必須是 numpy.ndarray，目前是 {type(img)}")

    """
    繪製 bbox 和文字標籤。

    Parameters:
        img: 原始影像（np.ndarray）
        box: bbox 座標 (x1, y1, x2, y2)
        label: 要顯示的文字
        color: RGB 顏色 (tuple)

    Returns:
        img: 處理後影像（np.ndarray）
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def draw_box_tracks(img, box, label, color, track_id, track_history, track_time_history , max_len=50):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img 必須是 numpy.ndarray，目前是 {type(img)}")
    
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 加入目前中心點
    center_x = (x1 + x2) / 2
    img_center_x = img.shape[1] / 2
    center_x = 0.6 * center_x + 0.4 * img_center_x  # 向中心靠近
    center_y = (y1 + y2) / 2

    # 更新位置歷史
    track_history[track_id].append((center_x, center_y))

    # 保留最多 max_len 筆資料
    if len(track_history[track_id]) > max_len:
        track_history[track_id] = track_history[track_id][-max_len:]

    # 更新時間歷史     
    now = time.time() * 1000  # 轉為毫秒
    track_time_history[track_id].append(now)
    if len(track_time_history[track_id]) > max_len:
        track_time_history[track_id] = track_time_history[track_id][-max_len:]

    # 計算速度
    speed = compute_speed(track_history[track_id], track_time_history[track_id])  # pixels/ms

    # 畫移動軌跡
    points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
    if len(points) >= 2:
        cv2.polylines(img, [points], isClosed=False, color=(0, 255, 0), thickness=2)

    # 顯示速度文字
    speed_text = f"{speed:.2f} px/ms"
    cv2.putText(img, speed_text, (x1, min(img.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def compute_speed(history, time_stamps, window=5):
    if len(history) < window or len(time_stamps) < window:
        return 0.0

    _, y_old = history[-window]
    _, y_new = history[-1]
    dy = y_new - y_old

    t_old = time_stamps[-window]
    t_new = time_stamps[-1]
    dt = t_new - t_old

    if dt == 0:
        return 0.0

    return dy / dt  # 垂直位移 / 時間