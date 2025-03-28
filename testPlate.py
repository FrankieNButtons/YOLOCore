import torch
import cv2
import os
from ultralytics.nn.tasks import attempt_load_weights
from plate_recognition.plate_rec import get_plate_result, init_model

def load_model(weights, device):
    return attempt_load_weights(weights, device=device)

def process_image(img, detect_model, plate_rec_model, device):
    predict = detect_model(img)[0]
    outputs = predict.squeeze(0).permute(1, 0)  # 处理检测结果
    results = []
    for output in outputs:
        rect = [int(x) for x in output[:4].cpu().numpy()]
        roi_img = img[rect[1]:rect[3], rect[0]:rect[2]]
        plate_number, _, plate_color, _ = get_plate_result(roi_img, device, plate_rec_model, is_color=True)
        results.append((plate_number, plate_color, rect))
    return results

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")  
    detect_model = load_model('weights/yolov8s.pt', device)
    plate_rec_model = init_model(device, 'weights/plate_rec_color.pth', is_color=True)
    
    image_path = 'imames/carb.jpg'
    img = cv2.imread(image_path)
    results = process_image(img, detect_model, plate_rec_model, device)
    
    for plate_number, plate_color, rect in results:
        print(f"Plate: {plate_number}, Color: {plate_color}, Location: {rect}")
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
    
    cv2.imwrite('example_result.jpg', img)

if __name__ == "__main__":
    main()
