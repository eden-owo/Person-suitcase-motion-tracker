# run_rtsp.py

import threading
import argparse
import sys
sys.path.insert(0, '/home/eden/opencv/opencv-4.10.0/build_cuda/lib/python3')  # 根據你的實際路徑調整
# sys.path.insert(0, '/home/user/opencv/opencv/build_cuda/lib/python3')  # 根據你的實際路徑調整

import cv2
import time
import os
from collections import defaultdict
import platform
import numpy as np
from pathlib import Path
 
from utils.transform import RP
from utils.visualize import draw_box_and_mask
from utils.video_utils import load_video, get_video_properties, GpuResizer

import ultralytics.utils.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml

import queue 
q = queue.Queue(maxsize=20)  # 定義 queue，最大容量可依需求調整

from flask import Flask, Response, stream_with_context
app = Flask(__name__)

@app.route('/')
def index():
    return "✅ Flask 正常運作"

@app.route('/fall')
def fall():
    return Response(stream_with_context(generate_stream()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response("這裡是影片串流內容")

def start_flask(port_args):
    print("🚀 Flask 開始運行在 http://0.0.0.0:5001/")
    app.run(host='0.0.0.0', port=port_args)
    
def generate_stream():     
    while True:
        try:
            if latest_frame is None:
                time.sleep(0.01)
                continue

            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Stream Error] {e}")
            time.sleep(0.1)

def is_jetson():
    return (
        platform.machine() == 'aarch64' and
        (os.path.exists('/etc/nv_tegra_release') or os.path.exists('/etc/nvidia-container-runtime'))
    )

def Receive(args, width, height, fps, resize_size, video, gpu_resizer):
    print("start Receive")  

    while True:
        ret, frame = video.read() 
        if not ret:
            # print("⚠️ 無法讀取 frame，跳過")
            time.sleep(0.01)
            continue
        try:
            
            frame_resized = gpu_resizer.resize(frame, resize_size)
            # frame_resized = cv2.resize(frame, resize_size)

            # 如果 queue 滿了，就丟掉舊的 frame（保留最新的）
            try:
                q.get_nowait()  # 確保先丟最舊的，不會阻塞
            except queue.Empty:
                pass
            q.put_nowait(frame_resized)
            # print("📥 Frame 放入 Queue")

        except cv2.error as e:
            print(f"❌ Resize 發生錯誤: {e}")
        except Exception as e:
            print(f"❌ 其他錯誤: {e}")
       

def Display(args, width, height, fps, M, max_width, max_height, resize_size):
    global latest_frame
    if args.export:
        if not args.model.endswith(".pt"):  
            raise NotImplementedError
        pt_model = YOLO(args.model)        
        pt_model.export(format="engine", device=0, half=True, dynamic=False, int8=True)

        return
    else:
        if args.model.endswith(".pt") and args.export is not True:       
            from yolo.yolo_seg_onnx import YOLOv8Seg_onnx 
            from utils.segmentor import process_frame
            model = YOLO(args.model)
        elif args.model.endswith(".pt") and args.export:
            from yolo.yolo_seg_trt import YOLOv8Seg_TRT    
            from utils.segmentor_trt import process_frame        
            model = YOLO(args.model)          
        elif args.model.endswith(".engine"):
            if is_jetson():
                print("Jetson device detected.")
            from yolo.yolo_seg_trt import YOLOv8Seg_TRT
            from utils.segmentor_trt import process_frame
            model = YOLO(args.model)          
  
        elif args.model.endswith(".onnx"):
            from yolo.yolo_seg_onnx import YOLOv8Seg_onnx
            from utils.segmentor import process_frame
            model = YOLOv8Seg_onnx(args.model, args.conf, args.iou)
        else: 
            raise NotImplementedError  
   
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs("test", exist_ok=True)
        out = cv2.VideoWriter("test/output.mp4", fourcc, fps, (int(max_width), int(max_height)))
    else:
        out = None

    print("Start Displaying")

    allowed_classes = {28}
    colors = {28: (0, 255, 0)}  # suitcase
 
    # Store the track history
    track_history = defaultdict(lambda: [])
    track_time_history = defaultdict(list)
    track_box_history = defaultdict(list)
    total_FPS = total_frame = 0
    output = None
    
    dummy_frame = np.zeros((resize_size[1], resize_size[0], 3), dtype=np.uint8)
    _ = process_frame(model, dummy_frame, M, max_width, max_height, colors,
                track_history, track_time_history, track_box_history, allowed_classes)    
                
    if args.web:
        flask_thread = threading.Thread(target=start_flask, args=(args.port,))
        flask_thread.daemon = True    
        flask_thread.start()
    
    while True:
        if not q.empty():
            frame = q.get_nowait()
            start_time = time.time()
            output = process_frame(model, frame, M, max_width, max_height, colors,
                        track_history, track_time_history, track_box_history, allowed_classes)
            
            duration = time.time() - start_time
            FPS = 1.0 / duration if duration > 0 else 0
            
            total_FPS += FPS
            total_frame += 1
            # print(f"Frame latency: {latency_ms:.2f} ms")
            print(f"FPS: {FPS:.2f} | Avg FPS: {total_FPS / total_frame:.2f} | {type(model)}", end='\r')
            if args.view:
                cv2.imshow("Segmented Image", output)
            if args.web:
                latest_frame = output
           
        if out:            
            if output is not None and output.size > 0:    
                out.write(output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if out: out.release()
    cv2.destroyAllWindows()

def run_rtsp(args):
    RTSP_FILE = os.getenv("RTSP_FILE") or args.rtsp_file or "/workspace/Person-suitcase-motion-tracker/rtsp.txt"
    with open(RTSP_FILE, "r", encoding="utf-8") as f:
        RTSP = f.read().strip()

    if is_jetson():
        print("Jetson device detected.")
        gst_pipeline = (
            f"rtspsrc location={RTSP} latency=100 drop-on-latency=true ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=true max-buffers=4 sync=true"
        )

        video = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not video.isOpened():
            raise RuntimeError(f"Failed to open video stream: {RTSP}")
    else:
        video = cv2.VideoCapture(RTSP)

    width, height, fps = get_video_properties(video)

    if width == 0 or height == 0:
        raise ValueError(f"Invalid video dimensions: width={width}, height={height}")

    ret, frame = video.read()
    if not ret or frame is None:
        attempts = 0
        while attempts < 10:
            ret, frame = video.read()
            if ret and frame is not None:
                break
            print("⚠️ Frame 讀取失敗，重試中…")
            time.sleep(0.2)
            attempts += 1
        else:
            raise RuntimeError("❌ 無法讀取首幀")

    # 輸出影片設定（請根據resize調整尺寸，要特別注意尺寸是 (width, height)）
    resize_size = (int(width * args.resize_ratio), int(height * args.resize_ratio))
    
    # Upload to GPU and resize      
    gpu_resizer = GpuResizer()
    frame_resized = gpu_resizer.resize(frame, resize_size)
    # frame_resized = cv2.resize(frame, resize_size)

    # 使用者選點並取得矯正圖與原始四點
    # M = RP.photo_PR_roi(frame_resized)
    ## 建立已封裝物件
    M, max_width, max_height = RP(args.transform).photo_PR_roi(frame_resized)
    
    p1 = threading.Thread(target=Receive, args=(args, width, height, fps, resize_size, video, gpu_resizer))
    p2 = threading.Thread(target=Display, args=(args, width, height, fps, M, max_width, max_height, resize_size))
    
    try: 
        if args.export:
            p1.start()      
            p1.join()      

        else: 
            p1.start()   
            p2.start()
            p1.join()
            p2.join()
    except KeyboardInterrupt:
        print("\n⛔️ 手動中斷程式 (Ctrl+C)")
    finally:
            print("🧹 清理資源並退出")
            video.release()
            cv2.destroyAllWindows()
