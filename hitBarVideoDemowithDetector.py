from hitBar import hitBar;
import cv2;
from detector import Detector;
import numpy as np;
import torch;
from typing import List, Dict, Any, Optional, Tuple;


detector = Detector("./weights/bestforproblem20.pt");
hb1 = hitBar(name="hitBar1", imgSize=(0, 320), startPoint=(620, 70), endPoint=(620,200), monitor=["person", "car", "bus", "van"], width=10.0, maxHis=50, visualize=True);
hb2 = hitBar(name="hitBar2", imgSize=(0, 320), startPoint=(350, 70), endPoint=(225,125), monitor=["person", "car", "bus", "truck", "van"], width=15.0, maxHis=50, visualize=True);
hb1._monitor(["truck"]);

video = cv2.VideoCapture("./videos/clip1.mp4");
fps = video.get(cv2.CAP_PROP_FPS);
while True:
    ret, frame = video.read();
    if ret:
        img, detailedResult, hitBarResults= detector.detect(frame, 
                                              addingConf=False, 
                                              hitBars=[hb1, hb2], 
                                              verbosity=2);
        cv2.imshow("HitBar", img);
        if hitBarResults[0]["hitDetails"]:
            print(hitBarResults[0]);
        
        key = cv2.waitKey(int(1000 / fps));
        if key == ord("q"):
            break;
cv2.destroyAllWindows()
        
        
        