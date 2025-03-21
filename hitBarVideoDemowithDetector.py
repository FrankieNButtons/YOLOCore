from hitBar import hitBar;
import cv2;
from detector import Detector;
import numpy as np;
from typing import List, Dict, Any, Optional, Tuple;

detector = Detector("./weights/yolov8x.pt");
hb1 = hitBar(name="hitBar1", imgSize=(0, 320), startPoint=(620,200), endPoint=(620, 70), monitor=["person", "car", "bus"], width=10.0, maxLength=50, visualize=False);
hb2 = hitBar(name="hitBar2", imgSize=(0, 320), startPoint=(225,125), endPoint=(350, 70), monitor=["person", "car", "bus", "truck"], width=15.0, maxLength=50, visualize=False);
hb1._monitor(["truck"]);

video = cv2.VideoCapture("./videos/clip1.mp4");
fps = video.get(cv2.CAP_PROP_FPS);
while True:
    ret, frame = video.read();
    if ret:
        img, detailedResult, hitBarResults= detector.detect(frame, 
                                              addingConf=False, 
                                              hitBars=[hb1, hb2], 
                                              verbosity=0);
        cv2.imshow("HitBar", img);
        
        key = cv2.waitKey(int(1000 / fps));
        if key == ord("q"):
            break;
cv2.destroyAllWindows()
        
        
        