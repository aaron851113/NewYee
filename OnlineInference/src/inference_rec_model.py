import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import numpy as np
import onnxruntime as rt
import cv2 
from torchvision import transforms
from PIL import Image
from src.utils_rec import *

device = torch.device("cpu")

def sort_word(list1, list2): 
    mapped_pairs = list(map(list,zip(list2, list1)))               
    for t in range(0, len(mapped_pairs)-1):
        for i in range(0, len(mapped_pairs)-t):
            if i != len(mapped_pairs)-1 and mapped_pairs[i][0][0]>mapped_pairs[i+1][0][0]:
                temp = mapped_pairs[i]
                mapped_pairs[i] = mapped_pairs[i+1]
                mapped_pairs[i+1] = temp
            else:
                continue
    
    return mapped_pairs


def load_yolo_config(path):
    # data format: feats_name anchor_size anchor1_x anchor1_y anchor2_x anchor2_y ...
    anchors = []
    with open(path) as file:
        for text in file.read().splitlines():
            text = text.split(' ')
            out_feats = text[0]
            anchor_size = int(text[1])
            anchor = text[2:(2+2*anchor_size)]
            anchor = [ ( int(anchor[2*i]), int(anchor[2*i+1]) ) for i in range(anchor_size)]
            anchors.append(anchor)

    return anchors

def detect_objects(out_feats, anchors, num_classes, img_dim):
    yolo_outputs = YOLO_postprocess(out_feats,anchors[0], num_classes, img_dim)
    box = non_max_suppression(yolo_outputs, 0.7, 0.4)
    if box[0] is None:
        return None, None, None
        
    det_boxes, _, det_scores, det_labels = torch.split(box, [4,1,1,1],dim=1)
    det_scores = det_scores.view(-1)
    det_labels = det_labels.view(-1)

    return det_boxes, det_labels, det_scores

def recognition_plate(plate_image):

    IMAGE_DIM = [160, 352] 
    # image pre-process
    preprocess_transforms = transforms.Compose([
        transforms.Resize((IMAGE_DIM[0], IMAGE_DIM[1])),
        transforms.ToTensor()])

    # load yolo anchor info
    anchor_path = './src/anchors_rec.txt'
    onnx_path = './src/models/cfgDW.onnx'
    anchors = load_yolo_config(anchor_path)
    num_class = 34

    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = [out.name for out in sess.get_outputs()]

    # pre-process
    original_image = Image.fromarray(plate_image)
    #original_image = original_image.convert('RGB')
    image = preprocess_transforms(original_image)
    image = image.unsqueeze(0)
    image = image.numpy()

    # onnxruntime 
    out_feats = sess.run(output_name, {input_name: image})
    out_feats = torch.FloatTensor(out_feats).to(device)

    # yolo-based post-process
    det_boxes, det_labels, det_scores = detect_objects(out_feats[0], anchors=anchors, num_classes=num_class, img_dim=IMAGE_DIM)
    print('det_boxes :',det_boxes,'det_labels :',det_labels)
    if det_labels is not None :
        det_boxes = det_boxes.to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        original_dims_y = original_dims[0,1::2] / float(IMAGE_DIM[0])
        original_dims_x = original_dims[0,0::2] / float(IMAGE_DIM[1])

        det_boxes[..., 0::2] = det_boxes[..., 0::2] * original_dims_x
        det_boxes[..., 1::2] = det_boxes[..., 1::2] * original_dims_y

        # sort det_boxes by x-axis
        sort_list = sort_word(det_labels,det_boxes)

        det_labels = []
        for i in sort_list :
            det_labels.append(int(i[1].tolist()))
        det_labels = torch.Tensor(det_labels)
        #print(det_labels)

        det_labels = det_labels.int()
        # Decode class integer labels
        det_labels = [Elan_od_rev_label_map[l] for l in det_labels.to('cuda').tolist()]

        car_plate = ''.join(det_labels)
        #print(car_plate)
        return car_plate
    else:
        return '0'