from detector import Detector;
import cv2;
import numpy as np;
import ultralytics;


detector = Detector("./weights/yolov8x.pt");
cap = cv2.VideoCapture("./videos/clip1.mp4");
fps = cap.get(cv2.CAP_PROP_FPS);
while cap.isOpened():
    ret, frame = cap.read();
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...");
        break;
    
    processedImg, detailedResult = detector.detect(frame);
    print("detailedResult:", detailedResult);

    cv2.imshow("Detected Image", processedImg);
    key = cv2.waitKey(int(1000 / fps));
    if key == ord("q"):
        break;
cv2.destroyAllWindows();