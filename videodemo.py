from detector import Detector;
import cv2;
import numpy as np;
import ultralytics;


detector = Detector("./weights/yolov8x.pt");
cap = cv2.VideoCapture("./videos/clip2.mp4");
fps = cap.get(cv2.CAP_PROP_FPS);
while cap.isOpened():
    ret, frame = cap.read();
    if not ret:
        print("Can't receive frame (Or stream end?). Exiting ...");
        break;
    
    processedImg, detailedResult, _ = detector.detect(frame, 
                                                   addingConf=False, 
                                                   verbosity=2);

    cv2.imshow("Detected Image", processedImg);
    key = cv2.waitKey(int(1000 / fps));
    if key == ord("q"):
        break;
cv2.destroyAllWindows();