# visualize.py

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import numpy as np

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