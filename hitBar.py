import numpy as np;
import cv2;
import torch;
from typing import List, Dict, Any, Optional, Tuple;

class hitBar:
    """
    **Description**
    A hit bar that attached to a detector to count the moving objects that hit the bar in a certain direction.
    
    **Properties**
    - `imgSize`: The size of the image that the hit bar is attached to.
    - `startPoint`: The start point of the hit bar. Defaultly (0, imgSize[1]/2).
    - `endPoint`: The end point of the hit bar. Defaultly (imgSize[0], imgSize[1]/2).
    - `history`: The histories of the former detailedResults from detector.
    - `monitoredCatagories`: The categories that the hit bar is monitoring.
    - `direction`: The detection direction of the hit bar.
    - `width`: The width of the hit bar. Defaultly 2.
    - `realmIn`: The latter realm of the hit bar in the image, calculated as (startPoint, startPoint, startPoint - width * direction, endPoint - width * direction).
    - `realmOut`: The former realm of the hit bar in the image, calculated as (endPoint, startPoint, endPoint + width * direction, endPoint + width * direction).
`
    
    **Methods**
    - `__init__`: Initialize the hit bar for a detector with a imgSize & 2 points of the bar.
    - `update`: Update the hit bar with a new detection result.
    - `monitor`: Monitor the hit bar with a list of categories.
    - `_inRealm`: Check if a point is in a realm.
    """
    def __init__(self, img:np.ndarray, startPoint: Tuple[int, int], endPoint: Tuple[int, int], monitor: List[str]=None, visualize: bool=True, width: int=2):
        """
        **Description**
        Initialize the hit bar for a detector with a imgSize & 2 points of the bar.
        
        **Params**
        - `imgSize`: The size of the image that the hit bar is attached to.
        - `startPoint`: The start point of the hit bar. Defaultly (0, imgSize[1]/2).
        - `endPoint`: The end point of the hit bar. Defaultly (imgSize[0], imgSize[1]/2).
        
        **Returns**
        None
        """
        # 默认为0,0到imgSize[0]的水平上行碰撞线
        self.imgSize: Tuple[int, int] = img.shape;
        if not startPoint or not endPoint:
            self.startPoint = (0, int(self.imgSize[1]/2));
            self.endPoint = (self.imgSize[0], int(self.imgSize[1]/2));
        else:
            self.startPoint: Tuple[int, int] = startPoint;
            self.endPoint: Tuple[int, int] = endPoint;
        
        self.width: int = width;
        # 计算法方向
        self.direction: np.ndarray = np.array([-(self.endPoint[1] - self.startPoint[1]), \
                                                self.endPoint[0] - self.startPoint[0]]);
        
        # 框定碰撞区域
        self.realmIn: np.nparray= np.array([(startPoint, endPoint), \
                                   [startPoint - width * self.direction, endPoint - width * self.direction]]);
        self.realmOut: np.ndarray= np.array([[startPoint + width * self.direction, endPoint + width * self.direction], \
                                    [endPoint, endPoint]]); 

        self.history: List[dict] = list();

        self.Accumulator: Dict [str, int] = dict();

    def monitor(categories: List[str]):
        """
        **Description**  
        Monitor the hit bar with a list of categories.
        
        **Params**  
        - `categories`: list[str], The categories to monitor.
        
        **Returns**  
        None
        """
        # 更新监控的对象类别
        for category in categories:
            self.Accumulator[category] = 0;
        self.monitoredCatagories = categories;

    def update(self, detailedResult: Dict[str, Any]):
        """
        **Description**
        Update the hit bar with a new detection result.
        
        **Params**
        - `detailedResult`: The detailed result of the detection.
        
        **Returns**
        modifiedResult: Dict[str, Any], The modified result of the detection.
        """
        # 检查是否是第一帧
        
        self.history.append(detailedResult);
        if len(history) > 30:
            history.pop(0);
    
            
        # 整理detailedResult中的数据
        for idx, pointLatter in enumerate(detailedResult["midPoints"]):
            tag = detailedResult["labels"][idx];
            if tag in self.monitoredCatagories:
            
                id = detailedResult["IDs"][idx] if (len(detailedResult["IDs"]) == len(detailedResult["labels"])) else idx;
                numInCatagory = [x[1] for x in detailedResult["numProjection"][tag] if x[0] == id][0];
        
        
        
        # 如果需要，绘制碰撞线和碰撞区
        if visualize:
            cv2.line(img, self.startPoint, self.endPoint, (0, 0, 255), 2);
            cv2.rectangle(img, self.realmIn[0], self.realmIn[1], (0, 255, 0), self.width);
            cv2.rectangle(img, self.realmOut[0], self.realmOut[1], (0, 0, 255), self.width);
            
            
        # 判断是否在检测线“入区”
        if self._inRealm(pointLatter, self.realmIn):
            self.Accumulator[tag] += 1;
            
            
            
            
    
    
    
    def _inRealm(point: Tuple[int, int], realm: np.ndarray):
        """
        **Description**  
        Check if a point is in a realm
        
        **Params**  
        - `point`: Tuple[int, int], The point to check.
        - `realm`: The realm to check.
        
        **Returns**  
        inRealm: bool, True if the point is in the realm, False otherwise.
        """
        # TODO
        
        
        
        


if __name__ == "__main__":
    from .detector import Detector;
    hb = hitBar();