import torch;
import time;
from ultralytics import YOLO;

m = YOLO("./weights/yolov8m.pt");
torch.set_default_device("mps");
time.sleep(5);
print(m.track("./image/crossing.jpg")[0].boxes);