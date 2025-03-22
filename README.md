# Documentation For YOLOv8Core
Inference Core based on YOLOv8 for competition
## Usage
1. Install requirements  
`pip3 install -r requirements.txt`

2. import `detector` from `YOLOv8Core` module  
`from YOLOv8Core.detector import Detector`

3. Create a detector object(With certain `model_path`)  
`detector = Detector(model_path='./weights/yolov8x.pt')`

4. Use `detector.detect` to detect image  
`detections = detector.detect(image)`

## Examples
1. Detect(or Track) with detector
```python
detector: Detector = Detector("./weights/yolov8s.pt");
img: np.ndarray = cv2.imread("./image/dog.jpeg");

processedImg, detailedResult , _ = detector.detect(img);
# print("detailedResult:", detailedResult);

cv2.imshow("Detected Image", processedImg);
cv2.waitKey(0);
cv2.destroyAllWindows();
```
2. Track Objects in a video and enumerate object in each category seperately
```python
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
        print("Can't receive frame (Or stream end?). Exiting ...");
        break;
    
    processedImg, detailedResult = detector.detect(frame, 
                                                   addingConf=False, 
                                                   verbosity=2);

    cv2.imshow("Detected Image", processedImg);
    key = cv2.waitKey(int(1000 / fps));
    if key == ord("q"):
        break;
cv2.destroyAllWindows();
```
3. Detector with `hitBar`s:
```python
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
```

## Detector
The Detector class encapsulates YOLOv8 inference logic and provides an method to track objects with `YOLO` & `BoT-SORT` in an image, optionally draw bounding boxes, labels, confidence scores, and object counts on the image.
### Properties
 - `SUPPORTTED_CATEGORIES`: List[str], the whitelisted categories for detection(Can be modified).
 - `outImg`: Optional[np.ndarray], The output image with bounding boxes, labels, etc. (if enabled).
 - `detailedResult`: Dict[str, Any], A dict containing detection data (counts, boxes, labels, confidence, etc.).
 - `detectedCounts`: Dict[str, int], A dict containing detected counts per label.
 - `detectedBoxes`: List[List[float, float,float, float]], A list of detected bounding boxes.
 - `detectedLabels`: List[str], A list of detected labels.
 - `dectectedConf`: List[float], A list of detected confidence scores.
 - `detectedIDs`: List[int], A list of tracking IDs.
 - `detectedMidPoints`: List[Tuple[float, float]], A list of detected midpoints.
 - `numProjection`: Dict[str, List[Tuple[int, int]]], A dict containing the number of projections per Category.
### Methods
 - `__init__`: Initializes the Detector object with the specified model path.
 - `detect`: Detects objects in an image using the YOLOv8 model.
 - `_resetDetector`: Resets the Detector object.
 - `_loadModel`: Loads the YOLOv8 model.

#### detect
##### Parameters(for main method `detector.detect`)
 - `oriImg`: The input image as a NumPy array (BGR color space).
 - `conf`: The confidence threshold for detections (default=0.25).
 - `addingBoxes`: Whether to draw bounding boxes on the output image.
 - `addingLabel`: Whether to draw label text (e.g., "car", "dog").
 - `addingConf`: Whether to append confidence score next to the label.
 - `addingCount`: Whether to append count index per label (e.g., "No.1").
 - `pallete`: Optional dict defining BGR color tuples for each label.
 - `verbosity`: The level of verbosity for logging(0, 1, 2, larger for simpler output in CLI, defaultly `0`).
##### Returns
 - `outImg`: the detected image.
 - `detailedResult`: Result of detection in detailed format.
 - ``

## hitBar
A plugin function for detector to enable customed-finer detection in detector.
### Properties
 - `imgOut`: The output image.
 - `imgSize`: The size of the reference image (height, width).
 - `startPoint`: The start point of the hit bar (x, y).
 - `endPoint`: The end point of the hit bar (x, y).
 - `direction`: The normal vector of the line from startPoint to endPoint.
 - `width`: The half-thickness (in pixels) for realmIn & realmOut.
 - `maxLength`: The maximum length of the history buffer. If exceeded, the oldest frame will be removed.
 - `realmIn`: The negative-side realm (4 points).
 - `realmOut`: The positive-side realm (4 points).
 - `name`: The name of the hit bar for debugging/logging.
 - `visualize`: Whether to draw the hit bar and realms in update method if an image is provided.
 - `history`: A list storing the previous frames' detection results.
 - `monitoredCatagories`: The categories that need to be counted or checked.
 - `Accumulator`: A dictionary counting the crossing events.
### Methods
 - `__init__`: Initialize the hit bar with geometry and optional visualization switch.
 - `_monitor`: Monitor the hit bar with a list of categories.
 - `update`: Update the hit bar with a new detection result (the main logic to check crossing).
 - `_hasIn`: Check if this target was in realmIn in a previous frame.
 - `_inRealm`: Internal method to check if a point is inside a 4-point polygon realm.

#### __init__
##### Parameters
 - `img`: np.ndarray, A reference image used mainly for shape or optional drawing.
 - `startPoint`: Tuple (x, y) for the start point of the bar. If None, defaults to (0, midRow).
 - `endPoint`: (x, y) for the end point of the bar. If None, defaults to (imgWidth, midRow).
 - `name`: The name of this hit bar.
 - `monitor`: Optional initial categories to be monitored.
 - `width`: The thickness in pixel used to build realmIn/realmOut on either side.
 - `visualize`: Whether we draw the bar & realms in `update` if an image is provided.