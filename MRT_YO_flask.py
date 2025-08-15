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
yolo_model = YOLO('yolo11m-pose.pt')

all_connect = [
    (5,7), (7,9),(6,8), (8,10),
    (5,11), (6,12), (5,6),(11,12),
    (11,13), (13,15), (12,14), (14,16)
]

hand_connect = [(5,7), (7,9),(6,8), (8,10)]
body_connect = [(5,11), (6,12), (5,6),(11,12)]
leg_connect = [(11,13), (13,15), (12,14), (14,16)]

app = Flask(__name__)

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
    while True:
        try:
            if output_frame is None:
                time.sleep(0.01)
                continue

            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Stream Error] {e}")
            time.sleep(0.1)

    
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
RTSP_FILE = os.getenv("RTSP_FILE", "/workspace/Person-suitcase-motion-tracker/rtsp.txt")  # 可用環境變數覆蓋路徑
with open(RTSP_FILE, "r", encoding="utf-8") as f:
    RTSP = f.read().strip()

# RTSP = r'rtsp://admin:Pass1234@192.168.1.200:554/stream0'
alarm_output = r'/workspace/pose/alarm/fall_detect.txt'

cap = cv2.VideoCapture(RTSP)
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

global output_frame

while cap.isOpened():
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if size_check == 0 :
        print ("screen width = ", original_width)
        print ("screen height = ", original_height)
        size_check = 1
    results = yolo_model(frame, verbose=False)

    # 警報處理
    # 警報樣態會刷新警報參數，否則讀幀時慢慢減少
    if alarm > 0:
        alarm = alarm - 1
    
    for i in range(len(results[0].boxes)):
        #調取result中boundingbox和keypoint
        box = results[0].boxes.xyxy[i].cpu().numpy()
        # box = results[0].boxes.xyxy[i]
        x1, y1, x2, y2 = map(int, box)
        
        keypoints = results[0].keypoints.xy[i].cpu().numpy()
        confidences = results[0].keypoints.conf[i].cpu().numpy()  # 每個關節的信心指數
        #賦予身體節點座標與是否有效的關聯性
        #判斷節點的有效性
        #因預設為(0,0)，所以只要是(0,0)就設為無效
        #j>4用來忽略頭部節點
        points = {}
        for j, (keypoint, conf) in enumerate(zip(keypoints, confidences)):

            if j > 4:
                x, y = int(keypoint[0]), int(keypoint[1])
                if conf < 0.6:  # 門檻值，可調整（0.3～0.5常見）
                    negate_point = 1
                else:
                    negate_point = 0

                if (x, y) == (0, 0):
                    negate_point = 1
                else:
                    negate_point = 0
                
                points[j] = (x, y, negate_point)
                #cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        #下面使用到的參數設定
        left_line = 0
        right_line = 0
        left_degree = 0
        right_degree = 0
        body_degree = 0
        alarm_sloping = 0
        alarm_knee_location = 0
        draw_lines = [] #要畫的線條資訊
        #依序取出標記對，若兩者皆為有效點再畫線
        #依序從all_connect取出座標對(a,b)
        #再從point{}中取出point[a]和point[b]
        #接著存給skp,ekp
        for connect_idx, (start_kp, end_kp) in enumerate(all_connect):
            if start_kp in points and end_kp in points:
                x_skp, y_skp, negate_skp = points[start_kp]
                x_ekp, y_ekp, negate_ekp = points[end_kp]                
                
                if (negate_skp == 0) and (negate_ekp == 0): 
                # if (x1 <= x_skp <= x2) and (x1 <= x_ekp <= x2) and (y1 <= y_skp <= y2) and (y1 <= y_ekp <= y2):
                #這行可以讓連線只存在於框內
                #框外的節點是模型預測的
                    
                    if (0 <= connect_idx <=3):
                        hand_color = (255, 0, 179)
                        hand_line_width =  3
                        draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), hand_color, hand_line_width))
                    
                    elif (4 <= connect_idx <=7):  
                        body_color = (0, 0, 255)
                        body_line_width = 3
                        draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), body_color, body_line_width))
                        
                        #身體左右線處理
                        if (connect_idx == 4):
                            if (negate_skp == 0) and (negate_ekp == 0):
                                o_point = (0,0)
                                p_point = ((x_skp-x_ekp),(y_ekp-y_skp))
                                left_line = 1
                                left_degree = (math.degrees(math.atan2(*p_point)))
                        
                        if (connect_idx == 5):
                            if (negate_skp == 0) and (negate_ekp == 0):
                                o_point = (0,0)
                                p_point = ((x_skp-x_ekp),(y_ekp-y_skp))
                                right_line = 1
                                right_degree = (math.degrees(math.atan2(*p_point)))
                    
                    elif (8 <= connect_idx <=11):
                        leg_color = (0, 77, 255)
                        leg_line_width = 3
                        draw_lines.append(((x_skp, y_skp), (x_ekp, y_ekp), leg_color, leg_line_width))
                        
                        if (connect_idx == 8) or (connect_idx == 10):
                            if (y_skp >= y_ekp):
                                alarm = alarm_keep_frame
                                alarm_knee_location = 1

        #計算角度
        if (left_line == 0) or (right_line == 0) :
            body_degree = abs(left_degree + right_degree)
        else :
            body_degree = ((abs(left_degree + right_degree))/2)

        if (body_degree >= detect_degree):
            alarm = alarm_keep_frame
            alarm_sloping = 1
            #print("Alarm_sloping :",f"{body_degree:.2f}")

        #畫框、標角度                       
        if alarm_knee_location == 1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for pt1, pt2, color, thickness in draw_lines:
                cv2.line(frame, pt1, pt2, color, thickness)
            cv2.putText(frame, f'degree: {body_degree:.2f}', (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        # if (alarm_sloping == 1):
            # cv2.putText(frame, f'degree: {body_degree:.2f}', (x1, y1),
            # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # else:
        #     cv2.putText(frame, f'degree: {body_degree:.2f}', (x1, y1),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # if (alarm_knee_location == 1):
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #print("Alarm_knee_location")
        # else:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #frame輸出、調整畫面大小、計算FPS、發警報
    h, w = frame.shape[:2]

    #if (original_width > 1920) or (original_height > 1080):
    if (original_width > 960) or (original_height > 540):
        resize_width = (w // 2)
        resize_height = (h // 2)
        resized_frame = cv2.resize(frame, (resize_width, resize_height))
        if (new_size_check == 0):
            print ("new width = ", resize_width)
            print ("new height = ", resize_height)
            new_size_check = 1
        
    else:
        resized_frame = cv2.resize(frame, (w, h))
    
    curr_time = time.time()
    #fps = 1 / (curr_time - prev_time) if prev_time else 0
    if prev_time:
        delta_time = (curr_time-prev_time)
        fps = (1 / delta_time)
    else:
        fps = 0
    prev_time = curr_time
    #print(f"FPS：{fps}")

    # cv2.putText(resized_frame, f'FPS: {fps:.2f}', (20, 40),
    # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if alarm > 0:
        if (tag_time <= 0):
            tag_time = alarm_txt_sec
            with open(alarm_output, "w", encoding="utf-8") as alarm_txt:
                alarm_txt.write('ALARM')
                print("output txt")
        else:
            tag_time = (tag_time - delta_time)
            #print("tag time = ", tag_time)

        # cv2.putText(resized_frame, f'ALARM', (20, 80),
        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    else:
        tag_time = (tag_time - delta_time)
        #print("tag time = ", tag_time)
    
    #cv2.imshow('PoseLandmarker (Tasks API)', resized_frame)
    output_frame = resized_frame
    if cv2.waitKey(1) & 0xFF == 27:
        print("End video")
        break

    #frame_idx += 1

cap.release()
cv2.destroyAllWindows()

