import cv2
import asyncio
import threading

# 初始化视频捕获
cap = cv2.VideoCapture("http://208.193.47.61/mjpg/video.mjpg")

# 存储帧的列表
frames = []

def fetch():
    """从摄像头获取帧"""
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧，可能是流中断")
        return None
    return frame

def generate_frame():
    """持续获取帧并存入 frames 列表"""
    while True:
        frame = fetch()
        if frame is not None:
            frames.append(frame)

def show():
    """从 frames 列表中取出帧并显示"""
    while True:
        if frames:
            frame = frames.pop(0)
            cv2.imshow("frame", frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def video_streamer():
    """异步函数，用于以 MJPEG 格式流式传输帧"""
    while True:
        if frames:
            frame = frames.pop(0)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.03)  # 控制帧率，防止 CPU 过载

# 启动获取帧的线程
threading.Thread(target=generate_frame, daemon=True).start()

# 运行显示函数
show()
