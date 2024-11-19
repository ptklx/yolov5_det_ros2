import time
import numpy as np
import cv2
import json
import aidlite
import random
import math
# import argparse
import os
OBJ_CLASS_NUM = 4

MODEL_SIZE = 640


anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]

current_p =os.path.dirname(os.path.abspath(__file__))

coco_class = [
     'tube', 'water', 'coke', 'fanta', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier']

coco_class_new = ['caning','water','coke','fanta']
def eqprocess(image, size1, size2):
    h, w, _ = image.shape
    mask = np.zeros((size1, size2, 3), dtype=np.float32)
    scale1 = h / size1
    scale2 = w / size2
    if scale1 > scale2:
        scale = scale1
    else:
        scale = scale2
    img = cv2.resize(image, (int(w / scale), int(h / scale)))
    mask[:int(h / scale), :int(w / scale), :] = img
    return mask, scale


def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box


def NMS(dets, scores, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def detect_postprocess(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    '''
    检测输出后处理
    prediction: aidlite模型预测输出
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    '''
    h, w, _ = img1shape
    valid_condidates = prediction[prediction[..., 4] > conf_thres]
    valid_condidates[:, 5:] *= valid_condidates[:, 4:5]
    valid_condidates[:, :4] = xywh2xyxy(valid_condidates[:, :4])

    max_det = 300
    max_wh = 7680
    max_nms = 30000
    valid_condidates[:, 4] = valid_condidates[:, 5:].max(1)
    valid_condidates[:, 5] = valid_condidates[:, 5:].argmax(1)
    sort_id = np.argsort(valid_condidates[:, 4])[::-1]
    valid_condidates = valid_condidates[sort_id[:max_nms]]
    boxes, scores = valid_condidates[:, :4] + valid_condidates[:, 5:6] * max_wh, valid_condidates[:, 4]
    index = NMS(boxes, scores, iou_thres)[:max_det]
    out_boxes = valid_condidates[index]
    clip_coords(out_boxes[:, :4], img0shape)
    out_boxes[:, :4] = xyxy2xywh(out_boxes[:, :4])
    print("检测到{}个区域".format(len(out_boxes)))
    return out_boxes


def detect_postprocess_new(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    # outputs = np.transpose(np.squeeze(output[0]))
    outputs = np.squeeze(prediction[0])
    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor =img0shape[1] / img1shape[1]
    y_factor = img0shape[0] / img1shape[0]

    # Iterate over each row in the outputs array
    for i in range(rows):
        confidence = outputs[i][4]
        # Extract the class scores from the current row
        # classes_scores = outputs[i][4:]

        # Find the maximum score among the class scores
        # max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if confidence >= 0.25:
            classes_scores = outputs[i][5:]
            max_score = np.amax(classes_scores)
            if max_score>conf_thres:
            
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    out_boxes=[]
    for i in indices:
        # Get the box, score, and class ID corresponding to the index
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        box.append(score)
        box.append(class_id)
        out_boxes.append(box)
    return out_boxes



def draw_detect_res(img, det_pred):
    '''
    检测结果绘制
    '''
    img = img.astype(np.uint8)
    color_step = int(255 / len(coco_class))
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])

        print(i + 1, [x1, y1, x2, y2], score, coco_class[cls_id])

        cv2.putText(img, f'{coco_class[cls_id]}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2 + x1, y2 + y1), (0, int(cls_id * color_step), int(255 - cls_id * color_step)),
                      thickness=2)

    return img

def draw_circle_res(img,point):
    radius=10
    color = (0,0,255)
    thickness=-1
    cv2.circle(img, (int(point[0]),int(point[1])), radius, color, thickness)
    return img

def left2right_point(det_pred,need_class=0,mid_soda_beer=[0,720]):
    center_p = []
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])
        print(i + 1, [x1, y1, x2, y2], score, coco_class[cls_id])
        if cls_id ==need_class:
            center_info = [x1+x2/2,y1+y2/2]
            center_p.append(center_info)
    if center_p==[]:
        return []
    sorted_x = sorted(center_p, key=lambda va: va[0])
    sorted_y = sorted(center_p, key=lambda va: va[1])
    return sorted_x[0]   #x轴从左到右

def get_soda_beer_mid(det_pred,methodx=True):
    #  'water', soda  beer
    center_soda = []
    center_beer = []
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])
        # print(i + 1, [x1, y1, x2, y2], score, coco_class[cls_id])
        if cls_id ==1: # soda
            center_info = [x1+x2/2,y1+y2/2]
            center_soda.append(center_info)
        if cls_id ==2: # beer
            center_info = [x1+x2/2,y1+y2/2]
            center_beer.append(center_info)
    if center_beer==[]:
        return [0,1280]
    if center_soda==[]:
        return [1,0]
    
    if methodx:
        sorted_s = sorted(center_soda, key=lambda va: va[0])
        sorted_b = sorted(center_beer, key=lambda va: va[0])
        if sorted_s[-1][0]<sorted_b[0][0] or sorted_s[0][0]<sorted_b[0][0]:
            midx = (sorted_s[-1][0]+sorted_b[0][0])/2
            return [0,midx]
        else:
            if sorted_b[-1][0]<sorted_s[0][0] or sorted_b[0][0]<sorted_s[0][0] :
                midx = (sorted_s[-1][0]+sorted_b[0][0])/2
                return [1,midx]
            else:
                midx = (sorted_s[0][0]+sorted_s[-1][0])/2
                return [0,midx]
    else:
        sorted_s = sorted(center_soda, key=lambda va: va[1])
        sorted_b = sorted(center_beer, key=lambda va: va[1])
        if sorted_s[-1][1]<sorted_b[0][1]or sorted_s[0][1]<sorted_b[0][1]:
            midy = (sorted_s[-1][1]+sorted_b[0][1])/2
            return [0,midy]
        else:
            if sorted_b[-1][1]<sorted_s[0][1]or sorted_b[0][1]<sorted_s[0][1] :
                midy = (sorted_s[-1][1]+sorted_b[0][1])/2
                return [1,midy]
            else:
                midy = (sorted_s[-1][1]+sorted_b[0][1])/2
                return [0,midy]



def middleorleft2right_point(det_pred,shape,need_class=0,mid_soda_beer=[0,720],soda_beer=''):
    middle_w = shape[1]/2   # 宽，
    middle_h = shape[0]/2   # 高
    center_p = []
    min_w=middle_w
    min_h=middle_h

    for i in range(len(det_pred)):
        x1, y1, w, h = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])
        print(i + 1, [x1, y1, w, h], score, coco_class[cls_id])
        if cls_id == need_class:
            center_info = [x1+w/2,y1+h/2, w, h]
            if soda_beer=='soda':
                if center_info[0]< mid_soda_beer[1] and mid_soda_beer[0]==0:
                    center_p.append(center_info)
                    if min_w>w:
                        min_w=w
                    if min_h>h:
                        min_h=h
                elif center_info[0]> mid_soda_beer[1] and mid_soda_beer[0]==1:
                    center_p.append(center_info)
                    if min_w>w:
                        min_w=w
                    if min_h>h:
                        min_h=h
            elif soda_beer=='beer':
                if center_info[0]> mid_soda_beer[1] and mid_soda_beer[0]==0:
                    center_p.append(center_info)
                    if min_w>w:
                        min_w=w
                    if min_h>h:
                        min_h=h
                elif center_info[0]< mid_soda_beer[1] and mid_soda_beer[0]==1:
                    center_p.append(center_info)
                    if min_w>w:
                        min_w=w
                    if min_h>h:
                        min_h=h
            else:
                center_p.append(center_info)
                if min_w>w:
                    min_w=w
                if min_h>h:
                    min_h=h
          
                
    if center_p==[]:
        return []
    sorted_y = sorted(center_p, key=lambda va: va[1],reverse=True)
    bottle_row_center=[]
    for c_p in sorted_y:
        if sorted_y[0][1]-min_h*1.5<c_p[1]:
            bottle_row_center.append(c_p)
        else:
            break
    sorted_x = sorted(bottle_row_center, key=lambda va: va[0])
    if sorted_x[0][0]>middle_w:
        return sorted_x[-1]
    else:
        return sorted_x[0]


def customized_filtration(img, det_pred):
    dels = []
    for i in range(len(det_pred)):
        x, y, w, h = [int(t) for t in det_pred[i][:4]]
        cls_id = int(det_pred[i][5])
        color = img[int(y+h/2),int(x+w/2),:]
        if x > 550 and y > 300:
            dels.append(i)
            continue

        if cls_id == 1 and np.max(np.abs(color - (20,125,30))) > 20:
            dels.append(i)
            continue

    det_pred = np.delete(det_pred, dels, axis=0)
    return det_pred


def customized_filtration_min_max(img, det_pred):
    middle_w = img.shape[1]/3   # 0.3宽，
    middle_h = img.shape[0]/3   # 0.3高
    det_pred_result = []
    for i in range(len(det_pred)):
        x, y, w, h = [int(t) for t in det_pred[i][:4]]
        if w>middle_w or h >middle_h:  # 排除过大的框
           continue
        if  w<middle_w/20 or h <middle_h/25: # 排除过小的框
            continue
        if x <8 or x>img.shape[1]-8:
           continue
        if y <8  or y>img.shape[0]-8:
            continue
        det_pred_result.append(det_pred[i])
        
    return det_pred_result


def pcl_lenghs(matrix,center=[2,2],radius=2):
    # 指定中心点和半径
    # center = (2, 2)  # 中心点 (行, 列)
    # radius = 1
    # 获取矩阵的形状
    rows, cols = matrix.shape
    # 计算在半径内的点
    mean_values = []

    # 加速
    tem_radius = radius+1
    r_start = center[0]-tem_radius if (center[0]-tem_radius)>0 else 0
    r_end = center[0]+tem_radius if (center[0]+tem_radius)< rows else rows
    c_start = center[1]-tem_radius if (center[1]-tem_radius)>0 else 0
    c_end = center[1]+tem_radius if (center[1]+tem_radius)>0 else cols

    for i in range(r_start,r_end):
        for j in range(c_start,c_end):
            # 计算到中心点的欧几里得距离
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mean_values.append(matrix[i, j])
    if mean_values==[]:
        return 0
    # 计算均值
    mean_value = np.mean(mean_values)
    return  int(mean_value)



def pcl_max_lenghs(matrix,point):
    cx,cy,h,w = point
    roct = matrix[int(cy-h /10 ):int(cy + h / 10 ), int(cx - w / 10):int(cx + w / 2 )]
    mean_v= np.max(roct)
    return mean_v

def get_mid_pos(box,depth_data,randnum):
    distance_list = []
    # mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    mid_pos = [box[0],box[1]]
    h,w = depth_data.shape
    # min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])) #确定深度搜索范围
    min_val=min(abs(box[2]-4),abs(box[3]-4))
    #print(box,)
    for i in range(randnum):
        bias = random.randint(-min_val//4, min_val//4)
        y_p = int(mid_pos[1] + bias)
        x_p = int(mid_pos[0] + bias)
        y_p = y_p if y_p>0 else 0
        y_p = h-1 if y_p>h-1 else y_p

        x_p = x_p if x_p>0 else 0
        x_p = w-1 if x_p>w-1 else x_p

        dist = depth_data[y_p, x_p]
        # dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
        # cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4] #冒泡排序+中值滤波
    #print(distance_list, np.mean(distance_list))
    dis_mean = np.mean(distance_list)
    print("dis_mean",dis_mean)
    if math.isnan(dis_mean):
        return 100000
    else:
        return int(dis_mean)



class Detect():
    # YOLOv5 Detect head for detection models
    def __init__(self, nc=80, anchors=(), stride=[], image_size=640):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.stride = stride
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid, self.anchor_grid = [0] * self.nl, [0] * self.nl
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        base_scale = image_size // 8
        for i in range(self.nl):
            self.grid[i], self.anchor_grid[i] = self._make_grid(base_scale // (2 ** i), base_scale // (2 ** i), i)

    def _make_grid(self, nx=20, ny=20, i=0):
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x)
        yv, xv = yv.T, xv.T
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = np.stack((xv, yv), 2)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid = np.repeat(grid, self.na, axis=1) - 0.5
        anchor_grid = self.anchors[i].reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.repeat(anchor_grid, repeats=ny, axis=2)
        anchor_grid = np.repeat(anchor_grid, repeats=nx, axis=3)
        return grid, anchor_grid

    def sigmoid(self, arr):
        return 1 / (1 + np.exp(-arr))

    def __call__(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self.sigmoid(x[i])
            y[..., 0:2] = (y[..., 0:2] * 2. + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, self.na * nx * ny, self.no))

        return np.concatenate(z, 1)

class yolov5det:
    def __init__(self,path):
        self.size = 640
        self.target_model = path #os.path.join(current_p,'./models/cutoff_yolov5n_w8a8.qnn216.ctx.bin')
        self.model_type = 'QNN'
        self.init_qnn()

    def init_qnn(self):
        
        config = aidlite.Config.create_instance()
        if config is None:
            print("Create config failed !")
            return False
        config.implement_type = aidlite.ImplementType.TYPE_LOCAL
        if self.model_type.lower()=="qnn":
            config.framework_type = aidlite.FrameworkType.TYPE_QNN
        elif self.model_type.lower()=="snpe2" or self.model_type.lower()=="snpe":
            config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
            
        config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
        config.is_quantify_model = 1
        
        model = aidlite.Model.create_instance(self.target_model)
        if model is None:
            print("Create model failed !")
            return False
        input_shapes = [[1, self.size, self.size, 3]]
        # self.output_shapes = [[1, 20, 20, 255], [1, 40, 40, 255], [1, 80, 80, 255]]
        # self.output_shapes = [[1, 20, 20, 21], [1, 40, 40, 21], [1, 80, 80, 21]]
        self.output_shapes = [[1, 20, 20, 27], [1, 40, 40, 27], [1, 80, 80, 27]]
        model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                                self.output_shapes, aidlite.DataType.TYPE_FLOAT32)

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
        if  self.interpreter is None:
            print("build_interpretper_from_model_and_config failed !")
            return None
        result =  self.interpreter.init()
        if result != 0:
            print(f"interpreter init failed !")
            return False
        result =  self.interpreter.load_model()
        if result != 0:
            print("interpreter load model failed !")
            return False
        print("detect model load success!")
    def preprocess(self,image):
        # Get the height and width of the input image
        self.img_height, self.img_width = image.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        # image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        image_data = image_data.astype(np.float32)
        # Return the preprocessed image data
        return image_data
    def __call__(self,frame):
        # 图片做等比缩放
        img_processed = np.copy(frame)
        aic_flag =False
        if aic_flag:
            [height, width, _] = img_processed.shape
            length = max((height, width))
            scale = length / self.size
            # ratio=[scale,scale]
            image = np.zeros((length, length, 3), np.uint8)
            image[0:height, 0:width] = img_processed
            img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_input=cv2.resize(img_input,(self.size,self.size))
            
            mean_data=[0, 0, 0]
            std_data=[255, 255, 255]
            img_input = (img_input-mean_data)/std_data  # HWC

            img_input = img_input.astype(np.float32)
        else:
            self.input_width=640
            self.input_height=640
            img_input = self.preprocess(img_processed)
        
        
        # qnn run
        # invoke_time=[]
        # for i in range(invoke_nums):
        result =  self.interpreter.set_input_tensor(0, img_input.data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
        
        t1=time.time()
        result =  self.interpreter.invoke()
        cost_time = (time.time()-t1)*1000
        # invoke_time.append(cost_time)
        
        if result != 0:
            print("interpreter set_input_tensor() failed")
        stride8 =  self.interpreter.get_output_tensor(0)
        stride16 =  self.interpreter.get_output_tensor(1)
        stride32 =  self.interpreter.get_output_tensor(2)
        print("=======================================")
        print(f"inference time {cost_time}")
        print("=======================================")
         ##  后处理
        stride = [8, 16, 32]
        yolo_head = Detect(OBJ_CLASS_NUM, anchors, stride, MODEL_SIZE)
        validCount0 = stride8.reshape(*self.output_shapes[2]).transpose(0, 3, 1, 2)
        validCount1 = stride16.reshape(*self.output_shapes[1]).transpose(0, 3, 1, 2)
        validCount2 = stride32.reshape(*self.output_shapes[0]).transpose(0, 3, 1, 2)
        pred = yolo_head([validCount0, validCount1, validCount2])
        if aic_flag:
            det_pred = detect_postprocess(pred, frame.shape, [MODEL_SIZE, MODEL_SIZE, 3], conf_thres=0.5, iou_thres=0.4)
            det_pred[np.isnan(det_pred)] = 0.0
            det_pred[:, :4] = det_pred[:, :4] * scale
        else: 
            det_pred = detect_postprocess_new(pred, frame.shape, [MODEL_SIZE, MODEL_SIZE, 3], conf_thres=0.5, iou_thres=0.4)


        return det_pred

    def destory(self):
        result =  self.interpreter.destory()



from yolov5_soda_beer_w import yolov5det as yolov5_soda_beer
os.environ['ADSP_LIBRARY_PATH'] = '/usr/local/lib/aidlux/aidlite/snpe2/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
def main():
    # imgs = os.path.join(current_p,"pt_color.png")
    imgs = os.path.join(current_p,"models/071.jpg")
    print("Start main ... ...")
  
    qnn_yolo = yolov5det(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models/cutoff_8550_det_object_detection_best_fp16.qnn216.ctx.bin'))
    qnn_yolo_soda =yolov5_soda_beer(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models/cutoff_8550_soda_beer_water_fp16.qnn216.ctx.bin'))
    need_class = 0  #   'water', soda  beer   # 'tube',=1 'water',=2 'coke',=3 'fenda'=4,   
    # image process
    frame = cv2.imread(imgs)
    preds=qnn_yolo(frame)

    if len(preds):
        det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
        res_img = draw_detect_res(frame, det_pred)
        if need_class==0:
            det_pred_soda=qnn_yolo_soda(frame)  
            mid_soda_beer_x = get_soda_beer_mid(det_pred_soda)
            one_point = middleorleft2right_point(det_pred,(frame.shape[0],frame.shape[1]),need_class=need_class,mid_soda_beer=mid_soda_beer_x,soda_beer='soda')  # x,y
        else:
            one_point = middleorleft2right_point(det_pred,(frame.shape[0],frame.shape[1]),need_class=need_class)
        if one_point==[]:
            print(" no choose object !")
        else:
        
            res_img = draw_circle_res(res_img,one_point)
            # cv2.imwrite("box_result.png", circle_frame)
            # mean_v =get_mid_pos(one_point,dframe,24)
            # tmp_str = "[{},{},{}]".format(int(one_point[0]),int(one_point[1]),mean_v)
            # result_str = result_str.replace("[0,0,0]",tmp_str)
            # result_str = "{},{},{}".format(int(one_point[0]),int(one_point[1]),mean_v)
            # print(result_str)
        save_path=os.path.join(current_p,"result.jpg")
        cv2.imwrite(save_path, res_img)   
       

    qnn_yolo.destory()
    return True



if __name__ == "__main__":
    main()

