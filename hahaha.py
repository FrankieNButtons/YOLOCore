import cv2;
import asyncio;
import threading;
from typing import Optional;
import numpy as np;

# 初始化视频捕获 / Initialize video capture
videoCapture = cv2.VideoCapture("http://208.193.47.61/mjpg/video.mjpg");

# 存储帧的列表 / Frame buffer
frameBuffer: list = [];

def fetchFrame() -> Optional[np.ndarray]:
    """
    **Description**  
    从摄像头获取一帧图像 / Fetch a frame from the camera

    **Params**  
    - None

    **Returns**  
    - `Optional[np.ndarray]`: 返回图像帧，失败时为 None / Frame or None if failed
    """
    ret, frame = videoCapture.read();
    if not ret:
        print("无法接收帧，可能是流中断 / Unable to receive frame, stream might be broken");
        return None;
    return frame;

def generateFrameLoop() -> None:
    """
    **Description**  
    持续从摄像头获取帧并加入帧列表 / Continuously fetch frames and add to buffer

    **Params**  
    - None

    **Returns**  
    - None
    """
    while True:
        frame = fetchFrame();
        if frame is not None:
            frameBuffer.append(frame);

def displayFrames() -> None:
    """
    **Description**  
    显示帧列表中的图像帧 / Display frames from buffer

    **Params**  
    - None

    **Returns**  
    - None
    """
    while True:
        if frameBuffer:
            frame = frameBuffer.pop(0);
            cv2.imshow("frame", frame);

        # 按 'q' 退出 / Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    videoCapture.release();
    cv2.destroyAllWindows();

async def videoStreamer() -> None:
    """
    **Description**  
    异步流式传输视频帧（MJPEG 格式）/ Stream video frames asynchronously (MJPEG)

    **Params**  
    - None

    **Returns**  
    - Async generator yielding JPEG-encoded frame bytes
    """
    while True:
        if frameBuffer:
            frame = frameBuffer.pop(0);
            _, buffer = cv2.imencode('.jpg', frame);
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n');
        await asyncio.sleep(0.03);  # 控制帧率，防止 CPU 过载 / Control FPS to avoid CPU overload

# 启动获取帧的线程 / Start frame acquisition thread
threading.Thread(target=generateFrameLoop, daemon=True).start();

# 启动显示线程 / Start frame display loop
displayFrames();
