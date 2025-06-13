# video_utils.py

import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2

def load_video(path: str):
    return cv2.VideoCapture(path)

def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, fps

def init_video_writer(output_path, size, fps, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, size)

def resize_frame_gpu(frame, size):
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(frame)
    resized_gpu = cv2.cuda.resize(gpu_mat, size)
    return resized_gpu.download()