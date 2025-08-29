#
#可使用之過渡版
#偵測合在畫骨架中
#未模組化
#未加入行李
#無限播放+產文字檔
#加入.cpu.numpy
#torch cuda已加
#

from flask import Flask, Response, stream_with_context
from ultralytics import YOLO
import threading
import cv2
import time
import math
import numpy as np
import time as time10
import os
import queue

cv2.setNumThreads(1)
app = Flask(__name__)

#共享佇列與鎖
frame_queue = queue.Queue(maxsize=2)  # 小佇列，避免堆積
output_frame = None
output_lock = threading.Lock()
stop_event = threading.Event()

yolo_model = YOLO('yolo11m-pose.pt')

all_connect = [
    (5,7), (7,9),(6,8), (8,10),
    (5,11), (6,12), (5,6),(11,12),
    (11,13), (13,15), (12,14), (14,16)
]

hand_connect = [(5,7), (7,9),(6,8), (8,10)]
body_connect = [(5,11), (6,12), (5,6),(11,12)]
leg_connect = [(11,13), (13,15), (12,14), (14,16)]

@app.route('/')
def index():
    return "✅ Flask 正常運作"

@app.route('/pose')
def pose():
    return Response(stream_with_context(generate_stream()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response("這裡是影片串流內容", mimetype='text/plain')
    
def start_flask():
    print("🚀 Flask 開始運行在 http://0.0.0.0:5000/")
    # app.run(host='0.0.0.0', port=5000, ssl_context=('192.168.1.22.pem', '192.168.1.22-key.pem'))
    app.run(host='0.0.0.0', port=5000)
    
def generate_stream():
    global output_frame
    while not stop_event.is_set():
        try:
            with output_lock:
                frame = None if output_frame is None else output_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Stream Error] {e}")
            time.sleep(0.05)  


#Thread A：RTSP 擷取（只負責抓幀，丟進佇列）
def capture_loop():
    """Thread A: RTSP 擷取 + Frame Queue"""
    cap = None
    while not stop_event.is_set():
        try:
            if cap is None or not cap.isOpened():
                print("[Capture] Opening RTSP...")
                cap = cv2.VideoCapture(RTSP)
                if not cap.isOpened():
                    print("[Capture] Open failed, retry in 1s")
                    time.sleep(1)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("[Capture] Read failed, reopen in 0.5s")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            # 丟掉舊幀：保持最新，降低延遲
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame)
        except Exception as e:
            print(f"[Capture Error] {e}")
            time.sleep(0.2)

    if cap is not None:
        cap.release()


#Thread B：YOLO 推論 + 畫圖（消耗佇列，產生 output_frame）
def infer_loop():
    """Thread B: YOLO 推論 + 繪圖（消耗 Queue）"""
    global output_frame, prev_time, size_check, new_size_check
    global alarm, tag_time, delta_time

    original_width = None
    original_height = None

    while not stop_event.is_set():
        try:
            # 取最新幀（若 1 秒沒幀就 loop 繼續）
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if original_width is None or original_height is None:
                h0, w0 = frame.shape[:2]
                original_width, original_height = w0, h0
                if size_check == 0:
                    print("screen width = ", original_width)
                    print("screen height = ", original_height)
                    size_check = 1

            # === 你的推論與骨架邏輯 開始 ===
            results = yolo_model(frame, verbose=False)

            if alarm > 0:
                alarm -= 1

            for i in range(len(results[0].boxes)):
                # bbox
                box = results[0].boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                keypoints = results[0].keypoints.xy[i].cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()

                points = {}
                for j, (keypoint, conf) in enumerate(zip(keypoints, confidences)):
                    if j > 4:
                        x, y = int(keypoint[0]), int(keypoint[1])
                        if conf < 0.6 or (x, y) == (0, 0):
                            negate_point = 1
                        else:
                            negate_point = 0
                        points[j] = (x, y, negate_point)

                left_line = 0; right_line = 0
                left_degree = 0; right_degree = 0
                body_degree = 0
                alarm_sloping = 0
                alarm_knee_location = 0
                draw_lines = []

                for connect_idx, (start_kp, end_kp) in enumerate(all_connect):
                    if start_kp in points and end_kp in points:
                        x_skp, y_skp, negate_skp = points[start_kp]
                        x_ekp, y_ekp, negate_ekp = points[end_kp]
                        if negate_skp == 0 and negate_ekp == 0:
                            if 0 <= connect_idx <= 3:
                                draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), (255, 0, 179), 3))
                            elif 4 <= connect_idx <= 7:
                                draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), (0, 0, 255), 3))
                                if connect_idx == 4:
                                    p_point = ((x_skp - x_ekp), (y_ekp - y_skp))
                                    left_line = 1
                                    left_degree = (math.degrees(math.atan2(*p_point)))
                                if connect_idx == 5:
                                    p_point = ((x_skp - x_ekp), (y_ekp - y_skp))
                                    right_line = 1
                                    right_degree = (math.degrees(math.atan2(*p_point)))
                            elif 8 <= connect_idx <= 11:
                                draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), (0, 77, 255), 3))
                                if connect_idx in (8, 10):
                                    if y_skp >= y_ekp:
                                        alarm = alarm_keep_frame
                                        alarm_knee_location = 1

                if (left_line == 0) or (right_line == 0):
                    body_degree = abs(left_degree + right_degree)
                else:
                    body_degree = (abs(left_degree + right_degree)) / 2

                if body_degree >= detect_degree:
                    alarm = alarm_keep_frame
                    alarm_sloping = 1

                if alarm_knee_location == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    for pt1, pt2, color, thickness in draw_lines:
                        cv2.line(frame, pt1, pt2, color, thickness)
                    cv2.putText(frame, f'degree: {body_degree:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # resize + FPS + ALARM 寫檔
            h, w = frame.shape[:2]
            if (original_width > 960) or (original_height > 540):
                resized_frame = cv2.resize(frame, (w // 2, h // 2))
                if new_size_check == 0:
                    print("new width = ", w // 2)
                    print("new height = ", h // 2)
                    new_size_check = 1
            else:
                resized_frame = cv2.resize(frame, (w, h))

            curr_time = time.time()
            if prev_time:
                delta_time = (curr_time - prev_time)
                fps = (1 / delta_time)
            else:
                fps = 0
            prev_time = curr_time

            if alarm > 0:
                if tag_time <= 0:
                    tag_time = alarm_txt_sec
                    os.makedirs(os.path.dirname(alarm_output), exist_ok=True)
                    with open(alarm_output, "w", encoding="utf-8") as alarm_txt:
                        alarm_txt.write('ALARM')
                    print("output txt")
                else:
                    tag_time -= delta_time
            else:
                tag_time -= delta_time

            # 更新共享輸出
            with output_lock:
                output_frame = resized_frame

            # 這裡不再 cv2.imshow / waitKey；顯示交給 Flask
            # === 你的推論與骨架邏輯 結束 ===

        except Exception as e:
            print(f"[Infer Error] {e}")
            time.sleep(0.05)


if __name__ == "__main__":

    #影片設定
    video_path_1 = r'E:\otoTim\MRTpose\testvideo\67sA18.mp4'
    video_path_2 = r'E:\otoTim\MRTpose\testvideo\short1p.mp4'
    video_path_3 = r'E:\otoTim\MRTpose\testvideo\A02_0604.avi'
    video_path_4 = r'E:\otoTim\MRTpose\testvideo\A21_0604.avi'
    video_path_5 = r'E:\otoTim\MRTpose\testvideo\A02_0604_2.mp4'
    video_path_6 = r'E:\otoTim\MRTpose\testvideo\FB_0605.mp4'
    video_path_7 = r'E:\otoTim\MRTpose\testvideo\FB_0606.mp4'
    video_path_8 = r'E:\otoTim\MRTpose\testvideo\A08_0606_1.mp4'
    video_path_9 = r'E:\otoTim\MRTpose\testvideo\A08_0606_2.mp4'
    video_path_10 = r'E:\otoTim\MRTpose\testvideo\A12_0606_1.mp4'
    video_path_11 = r'E:\otoTim\MRTpose\testvideo\A12_0606_2.mp4'
    video_path_12 = r'E:\otoTim\MRTpose\testvideo\A12_0606_3.mp4'
    video_path_13 = r'E:\otoTim\MRTpose\testvideo\TPE_0606_1.mp4'
    video_path_14 = r'E:\otoTim\MRTpose\testvideo\TPE_0606_2.mp4'
    video_path_15 = r'E:\otoTim\MRTpose\testvideo\FB_0609.mp4'

    # RTSP = r'rtsp://admin:Pass1234@192.168.1.102:554/stream0'
    RTSP_FILE = os.getenv("RTSP_FILE", "/workspace/Person-suitcase-motion-tracker/rtsp_cam1.txt")  # 可用環境變數覆蓋路徑
    with open(RTSP_FILE, "r", encoding="utf-8") as f:
        RTSP = f.read().strip()

    # RTSP = r'rtsp://admin:Pass1234@192.168.1.200:554/stream0'
    alarm_output = r'/workspace/pose/alarm/fall_detect.txt'

    prev_time = 0

    #設定使用變數
    alarm = 0
    alarm_keep_frame = 30
    alarm_txt_sec = 60
    tag_time = 0
    delta_time = 0
    size_check = 0
    new_size_check = 0
    detect_degree = 50

    flask_thread = threading.Thread(target=start_flask)
    flask_thread.daemon = True    
    flask_thread.start()

    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    infer_thread = threading.Thread(target=infer_loop, daemon=True)
    cap_thread.start()
    infer_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_event.set()