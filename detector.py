"""

This file contains the Detector class used for object detection via YOLOv8 models. 
It includes methods for model loading, image detection, and drawing bounding boxes with labels, confidence, and count.
"""

import cv2;
import numpy as np;
import seaborn as sns;
import matplotlib.pyplot as plt;
import ultralytics;
from ultralytics import solutions;
import os;
import logging;

from typing import Dict, List, Any, Tuple, Optional;
from ultralytics import YOLO;
from hitBar import hitBar;

# 环境与模型基本检查
ultralytics.checks();


class Detector:
    """
    
    **Description**  
    The Detector class encapsulates YOLOv8 inference logic and provides an interface
    to detect objects in an image, optionally draw bounding boxes, labels, confidence scores, and object counts on the image.

    **Properties**  
    - `SUPPORTTED_CATEGORIES`: List[str], the whitelisted categories for detection
    - `outImg`: Optional[np.ndarray], The output image with bounding boxes, labels, etc. (if enabled).
    - `detailedResult`: Dict[str, Any], A dict containing detection data (counts, boxes, labels, confidence, etc.).
    - `detectedCounts`: Dict[str, int], A dict containing detected counts per label.
    - `detectedBoxes`: List[List[float, float,float, float]], A list of detected bounding boxes.
    - `detectedLabels`: List[str], A list of detected labels.
    - `dectectedConf`: List[float], A list of detected confidence scores.
    - `detectedIDs`: List[int], A list of tracking IDs.
    - `detectedMidPoints`: List[Tuple[float, float]], A list of detected midpoints.
    - `numProjection`: Dict[str, List[Tuple[int, int]]], A dict containing the number of projections per Category.
    - `hitBarResults`: List[Dict[str, int]], A list containing hitBars' evenBetterResults for each hitBar.
    - `accidentBoxes`: List[Tuple[float, float, float, float]], A list of accident bounding boxes.
    - `accidentsConf`: List[float], A list of accident confidence scores.

    **Methods**
    - `__init__`: Initializes the Detector object with the specified model path.
    - `detect`: Detects objects in an image using the YOLOv8 model.
    - `_resetDetector`: Resets the Detector object.
    - `_loadModel`: Loads the YOLOv8 model.
    """

    SUPPORTTED_CATEGORIES: List[str] = ["person", "car", "bus", "van", "truck"];
    MAX_COUNT: int = 1000;

    def __init__(self, modelPath: str = "./weights/yolov8m.pt", accidentDetection: bool=True) -> None:
        """
        **Description**  
        Initializes the Detector object with a specified YOLOv8 model path.

        **Params**
        - `modelPath`: The file path for the YOLOv8 model weights.

        **Returns**
        - None
        """
        self._loadModel(modelPath, accidentDetection=accidentDetection);
        self.outImg: Optional[np.ndarray] = None;
        self.detailedResult: Dict[str, Any] = {
            "success": True,
            "count": dict(),
            "message": "Successfully detected"
        };
        self.detectedCounts: Dict[str, int] = dict();
        self.detectedBoxes: List[List[float, float,float, float]] = list();
        self.detectedLabels: List[str] = list();
        self.dectectedConf: List[float] = list();
        self.detectedIDs: List[int] = list();
        self.detectedMidPoints: List[Tuple[float, float]] = list();
        self.numProjection: Dict[str, List[Tuple[int, int]]] = dict();
        self.hitBarResults: List[Dict[str, int]] = list();
        self.accidentBoxes: List[Tuple[float, float, float, float]] = list();
        self.accidentConf: List[float] = list();
        
        
    def detect(
        self,
        oriImg: np.ndarray,
        conf: float = 0.25,
        addingBoxes: bool = True,
        addingLabel: bool = True,
        addingConf: bool = True,
        addingCount: bool = True,
        pallete: Optional[Dict[str, Tuple[int, int, int]]] = None,
        hitBars: Optional[List[hitBar]] = None,
        verbosity:int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        **Description**  
        The outer part of the detection process.
        """
        # 执行检测
        outImg, detailedResult, hitBarResult = self._detect(
            oriImg,
            conf,
            addingBoxes,
            addingLabel,
            addingConf,
            addingCount,
            pallete,
            hitBars,
            verbosity
        );
        
        # 释放资源,避免累积编号
        self._resetDetector();
        return outImg, detailedResult, hitBarResult;
        
    def _detect(
        self,
        oriImg: np.ndarray,
        conf: float,
        addingBoxes: bool,
        addingLabel: bool,
        addingConf: bool,
        addingCount: bool,
        pallete: Optional[Dict[str, Tuple[int, int, int]]],
        hitBars: Optional[List[hitBar]],
        verbosity: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        **Description**  
        Detect objects in an image using the loaded YOLOv8 model. Optionally draws 
        bounding boxes, labels, confidence scores, and detection counts on the image.

        **Params**
        - `oriImg`: np.ndarray, The input image as a NumPy array (BGR color space).
        - `conf`: float, The confidence threshold for detections (default=0.25).
        - `addingBoxes`: bool, Whether to draw bounding boxes on the output image.
        - `addingLabel`: bool, Whether to draw label text (e.g., "car", "dog").
        - `addingConf`: bool, Whether to append confidence score next to the label.
        - `addingCount`: bool, Whether to append count index per label (e.g., "No.1").
        - `pallete`: Dict[str, Tuple[int, int, int]], Optional dict defining BGR color tuples for each label.
        - `verbosity`: int, The level of verbosity for logging(0, 1, 2, larger for simpler output, defaultly `0`)

        **Returns**
        - `outImg`: The resultant image (with bounding boxes, labels, etc. if enabled).
        - `detailedResult`: A dict containing detection data (counts, boxes, labels, confidence, etc.).
        """
        if verbosity == 2:
            logging.getLogger("ultralytics").setLevel(logging.WARNING);
        
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
            if self.accDetector is not None:
                accidents = self.accDetector(oriImg);
            results = self.model.track(source=oriImg, conf=conf, persist=True);
        except Exception as e:
            print(f"Unable to process images due to:\n{e}");
            self.outImg = oriImg;
            self.detailedResult = {
                "success": False,
                "message": f"Unable to process images due to {e}"
            };
            return self.outImg, self.detailedResult;

        self.outImg = oriImg;

        if len(results) > 0 and results[0].boxes is not None:
            clsList = results[0].boxes.cls.cpu().numpy();

            self.detectedLabels = [results[0].names[int(clsIdx)] for clsIdx in clsList];
            self.detectedMidPoints = [results[0].boxes.xywh.cpu().numpy()[:, 0:2].tolist()];
            self.dectectedConf = results[0].boxes.conf.cpu().numpy().tolist();
            self.detectedBoxes = results[0].boxes.xyxy.cpu().numpy().tolist();
            self.accidentBoxes = accidents[0].boxes.xyxy.cpu().numpy().tolist();
            self.accidentConf = accidents[0].boxes.conf.cpu().numpy().tolist();
            
            
            if self.accidentConf is not None:
                for idx, confidence in enumerate(self.accidentConf):
                    
                    if confidence > 0.7:

                        cv2.rectangle(
                            self.outImg,
                            (int(self.accidentBoxes[idx][0]), int(self.accidentBoxes[idx][1])),
                            (int(self.accidentBoxes[idx][2]), int(self.accidentBoxes[idx][3])),
                            (0, 0, 255),
                            5
                        );
                        self.detailedResult["accidentBoxes"] = self.detailedResult.get("accidentBoxes", []);
                        self.detailedResult["accidentBoxes"].append(self.accidentBoxes[idx]);
                        
                        cv2.putText(
                            self.outImg,
                            f"ACCIDENT: {confidence:.2f}",
                            (int(self.accidentBoxes[idx][0]), int(self.accidentBoxes[idx][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2
                        );
                        self.detailedResult["accidentConf"] = self.detailedResult.get("accidentConf", []);
                        self.detailedResult["accidentConf"].append(self.accidentConf[idx]);
            
            
            try:
                self.detectedIDs = list(map(lambda x: int(x), results[0].boxes.id.cpu().numpy().tolist()));
            except Exception as e:
                print(f"IDs resetted as model didnot convolve in tracking.");
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
        self.detailedResult["IDs"] = self.detectedIDs;
        self.detailedResult["midPoints"] = self.detectedMidPoints[0];
        self.detailedResult["numProjection"] = self.numProjection;
        
        
        if bool(hitBars):
            for hb in hitBars:
                self.outImg, evenBetterResults = hb.update(self.outImg, self.detailedResult);
                self.hitBarResults.append(evenBetterResults);
        
        if verbosity == 0:
            print("DetailedResult:", self.detailedResult);
            print("hitBarResult:", self.hitBarResults);

        return self.outImg, self.detailedResult, self.hitBarResults;


    def _resetDetector(self) -> None:
        """
        **Description**  
        Resets the detection-related internal states of this Detector object.

        **Params**
        - None

        **Returns**
        - None
        """
        self.outImg: Optional[np.ndarray] = None;
        self.detailedResult: Dict[str, Any] = {
            "success": True,
            "count": dict(),
            "message": "Successfully detected"
        };
        self.detectedCounts: Dict[str, int] = dict();
        self.detectedBoxes: List[List[float, float,float, float]] = list();
        self.detectedLabels: List[str] = list();
        self.dectectedConf: List[float] = list();
        self.detectedIDs: List[int] = list();
        self.detectedMidPoints: List[Tuple[float, float]] = list();
        self.numProjection: Dict[str, List[Tuple[int, int]]] = dict();
        self.hitBarResults: List[Dict[str, int]] = list();
        self.accidentBoxes: List[Tuple[float, float, float, float]] = list();
        self.accidentConf: List[float] = list();



    def _loadModel(self, modelPath: str, accidentDetection: bool=True) -> None:
        """
        **Description**  
        Loads the YOLOv8 model from the given file path.

        **Params**
        - `modelPath`: The path to the YOLOv8 model weights.

        **Returns**
        - None
        """
        if accidentDetection:
            self.accDetector = YOLO("./weights/accdetect.pt");
        self.model = YOLO(modelPath);


if __name__ == "__main__":
    detector: Detector = Detector("./weights/yolov8s.pt");
    img: np.ndarray = cv2.imread("./image/dog.jpeg");
    
    processedImg, detailedResult , _ = detector.detect(img);
    # print("detailedResult:", detailedResult);

    cv2.imshow("Detected Image", processedImg);
    cv2.waitKey(0);
    cv2.destroyAllWindows();