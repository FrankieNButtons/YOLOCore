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
detector: Detector = Detector("./weights/yolov8x.pt");
img: np.ndarray = cv2.imread("./dog.jpeg");

processedImg, detailedResult = detector.detect(img);
print("detailedResult:", detailedResult);

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
 - `verbosity`: The level of verbosity for logging(0, 1, 2, larger for simpler output, defaultly `0`)
##### Returns
 - `outImg`: the detected image.
 - `detailedResult`: Result of detection in detailed format.

## hitBar
### Properties

### Methods