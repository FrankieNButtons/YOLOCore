from hitBar import hitBar;
import cv2;
from detector import Detector;
import numpy as np;
from typing import List, Dict, Any, Optional, Tuple;

h1his = [];
h2his = [];
detector = Detector("./weights/yolo12x.pt");
hb1 = hitBar(name="hitBar1", imgSize=(0, 320), startPoint=(225,125), endPoint=(350, 70), monitor=["person", "car", "bus"], width=20.0, maxLength=50, visualize=True);
hb2 = hitBar(name="hitBar2", imgSize=(0, 320), startPoint=(620,200), endPoint=(620, 70), monitor=["person", "car", "bus", "truck"], width=30.0, maxLength=50, visualize=True);
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
        h1his.append(hitBarResults[0]["hitDetails"]);
        if len(h1his) > 100:
            h1his.pop(0);
        h1Event = []
        for hitEvent in hitBarResults[1]["hitDetails"]:
            [h1Event.extend([h1Event["ID"] for h1Event in h1rec]) for h1rec in h1his]
        h2his.append(hitBarResults[1]["hitDetails"]);
        for event in hitBarResults[1]["hitDetails"]:
            if event["ID"] in h1Event:
                print(f"No.{event['numInCat']} {event['cat']} Passed from hitBar1 to hitBar2");
        cv2.imshow("Detect", img);
        
        key = cv2.waitKey(int(1000 / fps));
        if key == ord("q"):
            break;
cv2.destroyAllWindows()