import cv2;

# 直接使用 MJPEG 流 URL
url = "http://208.193.47.61/mjpg/video.mjpg";

# 通过 OpenCV 读取视频流
cap = cv2.VideoCapture(url);

if not cap.isOpened():
    print("无法打开视频流");
    exit();

while True:
    ret, frame = cap.read();
    if not ret:
        print("无法接收帧，可能是流中断");
        break;
    
    cv2.imshow("frame", frame);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cv2.destroyAllWindows();
    
    
    
    

# def fetch():
#     ret, frame = cap.read();
#     return frame

# def generate_frame():
#     while True:
#         frame = fetch()
#         yield frame
# frames = []

# def show():
#     for frame in generate_frame():
#         frames.append(frame)
     
        
        
# async def video_streamer():
#     while True:
#         frame = frames.pop(0)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', frame)[1].tobytes() + b'\r\n')
    
# cv2.imshow("frame", video_streamer())
        
        
# # while True:
    
# #     if not ret:
# #         print("无法接收帧，可能是流中断");
# #         break

# #     cv2.imshow("frame", frame)


# #     # 按 'q' 退出
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # 释放资源
# # cap.release()
# # cv2.destroyAllWindows()