from hitBar import hitBar;
import cv2;
from detector import Detector;
import numpy as np;
from typing import List, Dict, Any, Optional, Tuple;

detector = Detector("./weights/bestforproblem20.pt");
hitBar = hitBar(name="hitBarDemo", imgSize=(1920, 1080), startPoint=(800, 950), endPoint=(725,500), monitor=["person", "car", "bus"], width=50.0, maxHis=50, visualize=True);
hitBar._monitor(["truck"]);

video = cv2.VideoCapture("./videos/allCat.mp4");
fps = video.get(cv2.CAP_PROP_FPS);
while True:
    ret, frame = video.read();
    if ret:
        img, detailedResult, _ = detector.detect(frame, 
                                              addingConf=False, 
                                              verbosity=2);
        img, evenBetterResult = hitBar.update(img, detailedResult);
        cv2.imshow("HitBar", img);
        
        key = cv2.waitKey(int(1000 / fps));
        if key == ord("q"):
            break;
cv2.destroyAllWindows();