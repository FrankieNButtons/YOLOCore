import torch;
import cv2;
import numpy as np;
import copy;
import time;
import os;
from torch import nn;
import torch.nn.functional as F;
from ultralytics.nn.tasks import attempt_load_weights;

# Plate Recognition Part

PLATE_NAME = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";
COLOR = ['黑色', '蓝色', '绿色', '白色', '黄色'];
MEAN, STD = (0.588, 0.193);

class NetOCRColor(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, color_num=None):
        """
        **description** 
        Initializes the NetOCRColor model.

        **params** 
        cfg: Configuration for the model layers.
        num_classes: Number of output classes.
        export: Flag for exporting the model.
        color_num: Number of color classes.

        **returns** 
        None.
        """
        super(NetOCRColor, self).__init__();
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256];
        self.feature = self.make_layers(cfg, True);
        self.export = export;
        self.color_num = color_num;
        self.conv_out_num = 12;  #颜色第一个卷积层输出通道12
        if self.color_num:
            self.conv1 = nn.Conv2d(cfg[-1], self.conv_out_num, kernel_size=3, stride=2);
            self.bn1 = nn.BatchNorm2d(self.conv_out_num);
            self.relu1 = nn.ReLU(inplace=True);
            self.gap = nn.AdaptiveAvgPool2d(output_size=1);
            self.color_classifier = nn.Conv2d(self.conv_out_num, self.color_num, kernel_size=1, stride=1);
            self.color_bn = nn.BatchNorm2d(self.color_num);
            self.flatten = nn.Flatten();
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False);
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1);
    def make_layers(self, cfg, batch_norm=False):
        """
        **description** 
        Creates layers for the model based on the configuration.

        **params** 
        cfg: Configuration for the model layers.
        batch_norm: Flag for using batch normalization.

        **returns** 
        Sequential model layers.
        """
        layers = [];
        in_channels = 3;
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1);
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)];
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)];
                in_channels = cfg[i];
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)];
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1);
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)];
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)];
                    in_channels = cfg[i];
        return nn.Sequential(*layers);
    
    def forward(self, x):
        """
        **description** 
        Forward pass for the model.

        **params** 
        x: Input tensor.

        **returns** 
        Output tensor after the forward pass.
        """
        x = self.feature(x);
        if self.color_num:
            x_color = self.conv1(x);
            x_color = self.bn1(x_color);
            x_color = self.relu1(x_color);
            x_color = self.color_classifier(x_color);
            x_color = self.color_bn(x_color);
            x_color = self.gap(x_color);
            x_color = self.flatten(x_color);
        x = self.loc(x);
        x = self.newCnn(x);
        if self.export:
            conv = x.squeeze(2);  # b *512 * width
            conv = conv.transpose(2, 1);  # [w, b, c]
            if self.color_num:
                return conv, x_color;
            return conv;
        else:
            b, c, h, w = x.size();
            assert h == 1, "the height of conv must be 1";
            conv = x.squeeze(2);  # b *512 * width
            conv = conv.permute(2, 0, 1);  # [w, b, c]
            output = F.log_softmax(conv, dim=2);
            if self.color_num:
                return output, x_color;
            return output;

def initModel(device, model_path, is_color=False):
    """
    **description** 
    Initializes the model from the checkpoint.

    **params** 
    device: Device to load the model onto.
    model_path: Path to the model checkpoint.
    is_color: Flag to indicate if color classification is needed.

    **returns** 
    Initialized model.
    """
    checkpoint = torch.load(model_path, map_location=device);
    model_state = checkpoint['state_dict'];
    cfg = checkpoint['cfg'];
    color_classes = 0;
    if is_color:
        color_classes = 5;
    m = NetOCRColor(num_classes=len(PLATE_NAME), export=True, cfg=cfg, color_num=color_classes);
    m.load_state_dict(model_state, strict=False);
    m.to(device);
    m.eval();
    return m;

def imageProcessing(img, device):
    """
    **description** 
    Processes the input image for the model.

    **params** 
    img: Input image.
    device: Device to process the image on.

    **returns** 
    Processed image tensor.
    """
    img = cv2.resize(img, (168, 48));
    img = np.reshape(img, (48, 168, 3));

    img = img.astype(np.float32);
    img = (img / 255. - MEAN) / STD;
    img = img.transpose([2, 0, 1]);
    img = torch.from_numpy(img);

    img = img.to(device);
    img = img.view(1, *img.size());
    return img;

def decodePlate(preds):
    """
    **description** 
    Decodes the predicted plate characters.

    **params** 
    preds: Predicted characters.

    **returns** 
    List of new predictions and their indices.
    """
    pre = 0;
    new_preds = [];
    index = [];
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            new_preds.append(preds[i]);
            index.append(i);
        pre = preds[i];
    return new_preds, index;

def getPlateResult(img, device, model, is_color=False):
    """
    **description** 
    Gets the result of the plate recognition.

    **params** 
    img: Input image.
    device: Device to process the image on.
    model: The recognition model.
    is_color: Flag to indicate if color classification is needed.

    **returns** 
    Plate number and probabilities.
    """
    input = imageProcessing(img, device);
    if is_color:  #是否识别颜色
        preds, color_preds = model(input);
        color_preds = torch.softmax(color_preds, dim=-1);
        color_conf, color_index = torch.max(color_preds, dim=-1);
        color_conf = color_conf.item();
    else:
        preds = model(input);
    preds = torch.softmax(preds, dim=-1);
    prob, index = preds.max(dim=-1);
    index = index.view(-1).detach().cpu().numpy();
    prob = prob.view(-1).detach().cpu().numpy();
    
    new_preds, new_index = decodePlate(index);
    prob = prob[new_index];
    plate = "";
    for i in new_preds:
        plate += PLATE_NAME[i];
    if is_color:
        return plate, prob, COLOR[color_index], color_conf;  # 返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate, prob;

def getSplitMerge(img):
    """
    **description** 
    Merges the upper and lower parts of the image.

    **params** 
    img: Input image.

    **returns** 
    Merged image.
    """
    h, w, c = img.shape;
    img_upper = img[0:int(5 / 12 * h), :];
    img_lower = img[int(1 / 3 * h):, :];
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]));
    new_img = np.hstack((img_upper, img_lower));
    return new_img;

def allFilePath(rootPath, allFileList):
    """
    **description** 
    Collects all file paths in the given directory.

    **params** 
    rootPath: Root directory path.
    allFileList: List to store file paths.

    **returns** 
    None.
    """
    fileList = os.listdir(rootPath);
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFileList.append(os.path.join(rootPath, temp));
        else:
            allFilePath(os.path.join(rootPath, temp), allFileList);

def letterBox(img, size=(640, 640)):
    """
    **description** 
    Resizes the image while maintaining aspect ratio.

    **params** 
    img: Input image.
    size: Target size.

    **returns** 
    Resized image and resizing parameters.
    """
    h, w, _ = img.shape;
    r = min(size[0] / h, size[1] / w);
    new_h, new_w = int(h * r), int(w * r);
    new_img = cv2.resize(img, (new_w, new_h));
    left = int((size[1] - new_w) / 2);
    top = int((size[0] - new_h) / 2);   
    right = size[1] - left - new_w;
    bottom = size[0] - top - new_h; 
    img = cv2.copyMakeBorder(new_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114));
    return img, r, left, top;

def xywh2xyxy(det):
    """
    **description** 
    Converts bounding box format from (x, y, w, h) to (x1, y1, x2, y2).

    **params** 
    det: Input tensor of bounding boxes.

    **returns** 
    Converted bounding boxes.
    """
    y = det.clone();
    y[:, 0] = det[:, 0] - det[0:, 2] / 2;
    y[:, 1] = det[:, 1] - det[0:, 3] / 2;
    y[:, 2] = det[:, 0] + det[0:, 2] / 2;
    y[:, 3] = det[:, 1] + det[0:, 3] / 2;
    return y;

def myNums(dets, iou_thresh):  #nms操作
    """
    **description** 
    Applies Non-Maximum Suppression (NMS) on the detections.

    **params** 
    dets: Detections tensor.
    iou_thresh: IoU threshold for suppression.

    **returns** 
    List of indices to keep.
    """
    y = dets.clone();
    y_box_score = y[:, :5];
    index = torch.argsort(y_box_score[:, -1], descending=True);
    keep = [];
    while index.size()[0] > 0:
        i = index[0].item();
        keep.append(i);
        x1 = torch.maximum(y_box_score[i, 0], y_box_score[index[1:], 0]);
        y1 = torch.maximum(y_box_score[i, 1], y_box_score[index[1:], 1]);
        x2 = torch.minimum(y_box_score[i, 2], y_box_score[index[1:], 2]);
        y2 = torch.minimum(y_box_score[i, 3], y_box_score[index[1:], 3]);
        zero_ = torch.tensor(0).to(device);
        w = torch.maximum(zero_, x2 - x1);
        h = torch.maximum(zero_, y2 - y1);
        inter_area = w * h;
        nuion_area1 = (y_box_score[i, 2] - y_box_score[i, 0]) * (y_box_score[i, 3] - y_box_score[i, 1]);  #计算交集
        union_area2 = (y_box_score[index[1:], 2] - y_box_score[index[1:], 0]) * (y_box_score[index[1:], 3] - y_box_score[index[1:], 1]);  #计算并集

        iou = inter_area / (nuion_area1 + union_area2 - inter_area);  #计算iou
        
        idx = torch.where(iou <= iou_thresh)[0];   #保留iou小于iou_thresh的
        index = index[idx + 1];
    return keep;

def restoreBox(dets, r, left, top):  #坐标还原到原图上
    """
    **description**  
    Restores the bounding box coordinates to the original image.

    **params**  
    dets: Detections tensor.
    r: Resize ratio.
    left: Left padding.
    top: Top padding.

    **returns**  
    Restored detections.
    """
    dets[:, [0, 2]] = dets[:, [0, 2]] - left;
    dets[:, [1, 3]] = dets[:, [1, 3]] - top;
    dets[:, :4] /= r;
    return dets;

def postProcessing(prediction, conf, iou_thresh, r, left, top):  #后处理
    """
    **description** 
    Post-processes the predictions.

    **params** 
    prediction: Model predictions.
    conf: Confidence threshold.
    iou_thresh: IoU threshold for NMS.
    r: Resize ratio.
    left: Left padding.
    top: Top padding.

    **returns** 
    Processed predictions.
    """
    prediction = prediction.permute(0, 2, 1).squeeze(0);
    xc = prediction[:, 4:6].amax(1) > conf;  #过滤掉小于conf的框         
    x = prediction[xc];
    if not len(x):
        return [];
    boxes = x[:, :4];  #框
    boxes = xywh2xyxy(boxes);  #中心点 宽高 变为 左上 右下两个点
    score, index = torch.max(x[:, 4:6], dim=-1, keepdim=True);  #找出得分和所属类别
    x = torch.cat((boxes, score, x[:, 6:14], index), dim=1);  #重新组合
    
    score = x[:, 4];
    keep = myNums(x, iou_thresh);
    x = x[keep];
    x = restoreBox(x, r, left, top);
    return x;

def preProcessing(img, imSize, device):  #前处理
    """
    **description** 
    Pre-processes the image for detection.

    **params** 
    img: Input image.
    imSize: Image size for resizing.
    device: Device to process the image on.

    **returns** 
    Processed image and resizing parameters.
    """
    img, r, left, top = letterBox(img, (imSize, imSize));
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy();  #bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device);
    img = img.float();
    img = img / 255.0;
    img = img.unsqueeze(0);
    return img, r, left, top;

def detRecPlate(img, img_ori, detect_model, plate_rec_model, imSize):
    """
    **description** 
    Detects and recognizes the plate in the image.

    **params** 
    img: Input image.
    img_ori: Original image for drawing results.
    detect_model: Detection model.
    plate_rec_model: Plate recognition model.
    imSize: Image size for processing.

    **returns** 
    List of results for detected plates.
    """
    resultsList = [];
    img, r, left, top = preProcessing(img, imSize, device);
    predict = detect_model(img);               
    predict = predict[0];
    outputs = postProcessing(predict, 0.3, 0.5, r, left, top);
    for output in outputs:
        result_dict = {};
        output = output.squeeze().cpu().numpy().tolist();
        rect = output[:4];
        
        rect = [int(x) for x in rect];
        label = output[-1];
        roi_img = img_ori[rect[1]:rect[3], rect[0]:rect[2]];

        if int(label):
            roi_img = getSplitMerge(roi_img);
        plate_number, rec_prob, plate_color, color_conf = getPlateResult(roi_img, device, plate_rec_model, is_color=True);
        
        result_dict['plate_no'] = plate_number;              # 车牌号
        result_dict['plate_color'] = plate_color;            # 车牌颜色
        result_dict['rect'] = rect;                          # 车牌roi区域
        result_dict['detect_conf'] = output[4];              # 检测区域得分
        result_dict['roi_height'] = roi_img.shape[0];        # 车牌高度
        result_dict['color_conf'] = color_conf;              # 颜色得分
        result_dict['plate_type'] = int(label);              # 单双层 0单层 1双层
        resultsList.append(result_dict);
    return resultsList;

def drawResult(orgimg, dict_list, is_color=False):         # 车牌结果画出来
    """
    **description** 
    Draws the detection results on the image.

    **params** 
    orgimg: Original image.
    dict_list: List of results to draw.
    is_color: Flag to indicate if color information is included.

    **returns** 
    Image with drawn results.
    """
    result_str = "";
    for result in dict_list:
        rect_area = result['rect'];
        
        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1];
        padding_w = 0.05 * w;
        padding_h = 0.11 * h;
        rect_area[0] = max(0, int(x - padding_w));
        rect_area[1] = max(0, int(y - padding_h));
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w));
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h));

        height_area = result['roi_height'];
        result_p = result['plate_no'];
        if result['plate_type'] == 0:  #单层
            result_p += " " + result['plate_color'] + " 单层";
        else:  #双层
            result_p += " " + result['plate_color'] + " 双层";
        result_str += result_p + " ";
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2);  #画框
        
        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1);  #获得字体的大小
        cv2.putText(orgimg, result_p, 
                    (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])) + labelSize[1]),  # 调整y坐标确保文字居中
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,  # 字体缩放比例
                    (0, 0, 0),  # 字体颜色BGR
                    2);
               
    print(result_str);
    return orgimg;

if __name__ == "__main__":
    detectModelPath = './weights/sord.pt';
    recModelPath = "./weights/plate_rec_color.pth";
    imagePath = 'imgs';
    imSize = 640;
    savePath = 'result';
    device = torch.device("mps" if torch.mps.is_available() else "cpu");  

    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)];
    
    if not os.path.exists(savePath): 
        os.mkdir(savePath);

    detectModel = attempt_load_weights(detectModelPath, device=device);
    plateRecModel = initModel(device, recModelPath, is_color=True);
    detectModel.eval();
    fileList = [];
    allFilePath(imagePath, fileList);
    count = 0;
    timeSum = 0;
    time_begin = time.time();
    for pic_ in fileList:
        print(count, pic_, end=" ");
        time_b = time.time();
        img = cv2.imread(pic_);
        img_ori = copy.deepcopy(img);
        resultsList = detRecPlate(img, img_ori, detectModel, plateRecModel, imSize=imSize);
        print(resultsList);
        time_e = time.time();
        ori_img = drawResult(img, resultsList);
        img_name = os.path.basename(pic_);  
        saveImPath = os.path.join(savePath, img_name);
        cost = time_e - time_b;
        if count:
            timeSum += cost;
        count += 1;
        cv2.imwrite(saveImPath, ori_img);
    print(f"sumTime time is {time.time() - time_begin} s, average pic time is {timeSum / (len(fileList) - 1)}");