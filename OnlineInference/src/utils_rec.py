import torch
import torch.nn as nn

# Elan_od labels and color setting (Bus)
# Label map
#Elan_od_labels = ('b', 'c', 'm', 'y', 'p', 't', 'bi', 'yb', 'a', 'o')
Elan_od_labels = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
Elan_od_label_map = {k: v for v, k in enumerate(Elan_od_labels)}
#Elan_od_label_map['background'] = 0
Elan_od_rev_label_map = {v: k for k, v in Elan_od_label_map.items()}  # Inverse mapping

Elan_od_distinct_colors = [ (255, 0, 0), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 49), (145, 30, 180), (72, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),  (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75), (230, 25, 75),(0, 128, 128)]

Elan_od_label_color_map = {k: Elan_od_distinct_colors[i] for i, k in enumerate(Elan_od_label_map.keys())}


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    #print("prediction=",prediction)
    output = [None for _ in range(len(prediction))]
    #output = []
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        #print("image_pred=",image_pred)
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            #output[image_i] = torch.stack(keep_boxes)
            output = torch.stack(keep_boxes)

    return output

def compute_grid_offsets(grid_size, img_dim, anchors, cuda=True):
    
    num_anchors = len(anchors)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    stride_y = img_dim[0] / grid_size[0]
    stride_x = img_dim[1] / grid_size[1]

    # Calculate offsets for each grid
    grid_x = torch.arange(grid_size[1]).repeat(grid_size[0], 1).view([1, 1, grid_size[0], grid_size[1]]).type(FloatTensor)
    grid_y = torch.arange(grid_size[0]).repeat(grid_size[1], 1).t().view([1, 1, grid_size[0], grid_size[1]]).type(FloatTensor)
    scaled_anchors = FloatTensor([(a_w / stride_x, a_h / stride_y) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))

    return stride_x, stride_y, grid_x, grid_y, anchor_w, anchor_h

def YOLO_postprocess(feats, anchors, num_classes, img_dim):
    
    num_anchors = len(anchors)
    FloatTensor = torch.cuda.FloatTensor if feats.is_cuda else torch.FloatTensor

    # self.img_dim = img_dim
    num_samples = feats.size(0)
    grid_size = [feats.size(2), feats.size(3)]
    
    #1 5 39 195 5
    prediction = (
        feats.view(num_samples, num_anchors, num_classes+5, grid_size[0], grid_size[1])
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )

    # Get outputs
    x = torch.sigmoid(prediction[..., 0])  # Center x
    y = torch.sigmoid(prediction[..., 1])  # Center y
    w = prediction[..., 2]  # Width
    h = prediction[..., 3]  # Height
    pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
    pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

    # If grid size does not match current we compute new offsets
    stride_x, stride_y, grid_x, grid_y, anchor_w, anchor_h = compute_grid_offsets(grid_size, img_dim, anchors, cuda=x.is_cuda)
    
    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction[..., :4].shape)
    
    pred_boxes[..., 0] = (x.data + grid_x) * stride_x
    pred_boxes[..., 1] = (y.data + grid_y) * stride_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w * stride_x
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h * stride_y

    output = torch.cat(
        (
            pred_boxes.view(num_samples, -1, 4),
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, num_classes),
        ),
        -1,
    )

    return output

