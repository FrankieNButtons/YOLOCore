import numpy as np;
import cv2;
import torch;
from typing import List, Dict, Any, Optional, Tuple;

class hitBar:
    """
    **Description**
    A hit bar that attached to a detector to count the moving objects that hit the bar in a certain direction.
    
    **Params**
    ``:
    ``:
    ``:
    ``:
    
    **Methods**
    `__init__`: Initialize the hit bar for a detector with a imgSize & 2 points of the bar.
    """
    def __init__(self, imgSize: Tuple[int, int], startPoint: Tuple[int, int]=None, endPoint: Tuple[int, int]=None):
        """
        **Description**
        Initialize the hit bar for a detector with a imgSize & 2 points of the bar.
        
        **Params**
        
        
        **Returns**
        None
        """
        # 默认为0,0到imgSize的碰撞线
        self.imgSize: Tuple[int, int] = imgSize
        if not startPoint or not endPoint:
            self.startPoint = (0, int(imgSize[1]/2));
            self.endPoint = (imgSize[0], int(imgSize[1]/2));
        else:
            self.startPoint: Tuple[int, int] = startPoint;
            self.endPoint: Tuple[int, int] = endPoint;
        