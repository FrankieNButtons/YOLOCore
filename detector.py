"""

This file contains the Detector class used for object detection via YOLOv8 models. 
It includes methods for model loading, image detection, and drawing bounding boxes with labels, confidence, and count.
"""

import cv2;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import ultralytics;

from typing import Dict, List, Any, Tuple, Optional;
from ultralytics import YOLO;

# 环境与模型基本检查
ultralytics.checks();


class Detector:
    """
    **Description**
    The Detector class encapsulates YOLOv8 inference logic and provides an interface
    to detect objects in an image, optionally draw bounding boxes, labels, confidence scores, and object counts on the image.

    **Params**
    - SUPPORTTED_CATEGORIES: List[str], the whitelisted categories for detection
    - model: YOLO, the YOLO model instance
    - outImg: Optional[np.ndarray], the output image with drawn detections
    - detailedResult: Dict[str, Any], the dictionary containing detection stats
    - detectedCounts: Dict[str, int], record counts for each detected label
    - detectedBoxes: List[List[float]], bounding boxes [x1,y1,x2,y2] in pixel coords
    - detectedLabels: List[str], labels of each detection
    - dectectedConf: List[float], confidence for each detection

    **Methods**
    - `__init__`: Initializes the Detector object with the specified model path.
    - `detect`: Detects objects in an image using the YOLOv8 model.
    - `_resetDetector`: Resets the Detector object.
    - `_loadModel`: Loads the YOLOv8 model.
    """

    SUPPORTTED_CATEGORIES: List[str] = ["person", "car", "bus", "van", "truck"];
    MAX_COUNT: int = 100;

    def __init__(self, model_path: str = "./weights/yolov8m.pt") -> None:
        """
        **Description**
        Initializes the Detector object with a specified YOLOv8 model path.

        **Params**
        - `model_path`: The file path for the YOLOv8 model weights.

        **Returns**
        - None
        """
        self._loadModel(model_path);
        self.outImg: Optional[np.ndarray] = None;
        self.detailedResult: Dict[str, Any] = {
            "success": True,
            "count": dict(),
            "message": "Successfully detected"
        };
        self.detectedCounts: Dict[str, int] = dict();
        self.detectedBoxes: List[List[float]] = list();
        self.detectedLabels: List[str] = list();
        self.dectectedConf: List[float] = list();
        self.detectedIDs: List[int] = list();
        self.detectedMidPoints: List[Tuple[float, float]] = list();
        self.numProjection: Dict[str, List[Tuple[int, int]]] = dict();
        
        
    def detect(
        self,
        oriImg: np.ndarray,
        conf: float = 0.25,
        addingBoxes: bool = True,
        addingLabel: bool = True,
        addingConf: bool = True,
        addingCount: bool = True,
        pallete: Optional[Dict[str, Tuple[int, int, int]]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        **Description**
        The outer part of the detection process.
        """
        # 执行检测
        outImg, detailedResult = self._detect(
            oriImg,
            conf,
            addingBoxes,
            addingLabel,
            addingConf,
            addingCount,
            pallete
        );
        
        # 释放资源,避免累积编号
        self._resetDetector();
        return outImg, detailedResult;
        
    def _detect(
        self,
        oriImg: np.ndarray,
        conf: float,
        addingBoxes: bool,
        addingLabel: bool,
        addingConf: bool,
        addingCount: bool,
        pallete: Optional[Dict[str, Tuple[int, int, int]]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        **Description**
        Detect objects in an image using the loaded YOLOv8 model. Optionally draws 
        bounding boxes, labels, confidence scores, and detection counts on the image.

        **Params**
        - `oriImg`: The input image as a NumPy array (BGR color space).
        - `conf`: The confidence threshold for detections (default=0.25).
        - `addingBoxes`: Whether to draw bounding boxes on the output image.
        - `addingLabel`: Whether to draw label text (e.g., "car", "dog").
        - `addingConf`: Whether to append confidence score next to the label.
        - `addingCount`: Whether to append count index per label (e.g., "No.1").
        - `pallete`: Optional dict defining BGR color tuples for each label.

        **Returns**
        - `outImg`: The resultant image (with bounding boxes, labels, etc. if enabled).
        - `detailedResult`: A dict containing detection data (counts, boxes, labels, confidence, etc.).
        """
        if pallete is None:
            # 构建BGR调色板
            base_colors = sns.color_palette("bright", len(self.SUPPORTTED_CATEGORIES));
            pallete = {
                cat: (
                    int(base_colors[i][2] * 255),
                    int(base_colors[i][1] * 255),
                    int(base_colors[i][0] * 255)
                )
                for i, cat in enumerate(self.SUPPORTTED_CATEGORIES)
            };

        try:
            results = self.model.track(source=oriImg, conf=conf, persist=True);
        except Exception as e:
            print(f"Unable to process images due to:\n{e}");
            self.outImg = oriImg;
            self.detailedResult = {
                "success": False,
                "message": f"Unable to process images due to {e}"
            };
            return self.outImg, self.detailedResult;

        # 每次调用前重置检测结果
        self.outImg = oriImg;

        if len(results) > 0 and results[0].boxes is not None:
            cls_list = results[0].boxes.cls.cpu().numpy();
            conf_list = results[0].boxes.conf.cpu().numpy();
            box_list = results[0].boxes.xyxy.cpu().numpy();

            self.detectedLabels = [results[0].names[int(cls_idx)] for cls_idx in cls_list];
            self.detectedMidPoints = [results[0].boxes.xywh.cpu().numpy()[:, 0:2]];
            self.dectectedConf = conf_list.tolist();
            self.detectedBoxes = box_list.tolist();
            try:
                self.detectedIDs = list(map(lambda x: int(x), results[0].boxes.id.cpu().numpy().tolist()));
            except Exception as e:
                print(f"IDs resetted as model made wrong in tracking as {e}");
                self.detectedIDs = list(range(len(self.detectedLabels)));
                
                
                
            for idx, tag in enumerate(self.detectedLabels):
                self.detectedCounts[tag] = self.detectedCounts.get(tag, 0) + 1 % self.MAX_COUNT;
                self.numProjection[tag] = self.numProjection.get(tag, []);
                self.numProjection[tag].append((self.detectedIDs[idx], self.detectedCounts[tag]));
                
                
                box = self.detectedBoxes[idx];
                conf_val = self.dectectedConf[idx];
                if tag in self.SUPPORTTED_CATEGORIES:

                    if addingBoxes:
                        cv2.rectangle(
                            self.outImg,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            pallete[tag],
                            3
                        );

                    # 构建文本
                    labelString: str = "";
                    if addingCount:
                        num = [x[1] for x in self.numProjection[tag] if x[0] == self.detectedIDs[idx]][0];
                        labelString += f" No.{num} ";
                    if addingLabel:
                        labelString += tag.capitalize();
                    if addingConf:
                        labelString += f" {conf_val:.2f}";


                    # 绘制文本
                    if labelString:
                        cv2.putText(
                            self.outImg,
                            labelString,
                            (int(box[0]), int(box[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            pallete[tag],
                            2
                        );

            # 按支持类别统计总数
            self.detailedResult["count"] = {
                label: self.detectedLabels.count(label)
                for label in self.SUPPORTTED_CATEGORIES
            };

        
        self.detailedResult["boxes"] = self.detectedBoxes;
        self.detailedResult["labels"] = self.detectedLabels;
        self.detailedResult["confidence"] = self.dectectedConf;
        self.detailedResult["count"] = self.detectedCounts;
        
        

        return self.outImg, self.detailedResult;


    def _resetDetector(self) -> None:
        """
        **Description**
        Resets the detection-related internal states of this Detector object.

        **Params**
        - None

        **Returns**
        - None
        """
        self.outImg = None;
        self.detailedResult = {
            "success": True,
            "count": {},
            "message": "Successfully detected"
        };
        self.detectedCounts = {};
        self.detectedIDs = [];
        self.detectedBoxes = [];
        self.detectedLabels = [];
        self.dectectedConf = [];



    def _loadModel(self, model_path: str) -> None:
        """
        **Description**
        Loads the YOLOv8 model from the given file path.

        **Params**
        - `model_path`: The path to the YOLOv8 model weights.

        **Returns**
        - None
        """
        self.model = YOLO(model_path);


if __name__ == "__main__":
    detector: Detector = Detector("./weights/yolov8x.pt");
    img: np.ndarray = cv2.imread("./dog.jpeg");
    
    processedImg, detailedResult = detector.detect(img);
    print("detailedResult:", detailedResult);

    cv2.imshow("Detected Image", processedImg);
    cv2.waitKey(0);
    cv2.destroyAllWindows();