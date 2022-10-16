"""
iou计算函数
"""
import time

import torch
import torchvision

from Inferences.utils.tools import xywh2xyxy
from Inferences.utils.iou import box_iou, polygon_box_iou


# ## 下面是车网络nms
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape):
    # 转换网络输入size坐标到实际图片坐标
    coords[..., 0:4:2] = coords[..., 0:4:2] / img1_shape[1] * img0_shape[1]
    coords[..., 1:4:2] = coords[..., 1:4:2] / img1_shape[0] * img0_shape[0]
    return coords


# ## 下面是装甲板网络nms
def polygon_non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                                multi_label=False,
                                labels=(), max_det=300):
    """
        Runs Non-Maximum Suppression (NMS) on inference results for polygon boxes
        Returns:  list of detections, on (n,10) tensor per image [xyxyxyxy, conf, cls]
    """

    # prediction has the shape of (bs, all potential anchors, 89)
    assert not agnostic, "polygon does not support agnostic"
    nc = prediction.shape[2] - 9  # number of classes
    xc = prediction[..., 8] > conf_thres  # confidence candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 3, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into polygon_nms_kernel, can increase this value
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 10), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # xw = (x[..., 0:8:2].max(dim=-1)[0] - x[..., 0:8:2].min(dim=-1)[0]).view(-1, 1)
        # xh = (x[..., 1:8:2].max(dim=-1)[0] - x[..., 1:8:2].min(dim=-1)[0]).view(-1, 1)
        # x[((xw < min_wh) | (xw > max_wh) | (xh < min_wh) | (xh > max_wh)).any(1), 8] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 9), device=x.device)
            v[:, :8] = l[:, 1:9]  # box
            v[:, 8] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 9] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 9:] *= x[:, 8:9]  # conf = obj_conf * cls_conf

        # Box (x1, y1, x2, y2, x3, y3, x4, y4)
        box = x[:, :8].clone()

        # Detections matrix nx10 (xyxyxyxy, conf, cls)
        # Transfer sigmoid probabilities of classes (e.g. three classes [0.567, 0.907, 0.01]) to selected classes (1.0)
        if multi_label:
            i, j = (x[:, 9:] > conf_thres).nonzero(as_tuple=False).T
            # concat satisfied boxes (multi-label-enabled) along 0 dimension
            x = torch.cat((box[i], x[i, j + 9, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 9:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 9:10] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 8].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Polygon NMS does not support Batch NMS and Agnostic
        # x is the sorted predictions with boxes x[:, :8], confidence x[:, 8], class x[:, 9]
        # cannot use torchvision.ops.nms, which only deals with axis-aligned boxes
        i = polygon_nms_kernel(x, iou_thres)  # polygon-NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            boxes = x[:, :8]
            # update boxes as boxes(i,8) = weights(i,n) * polygon boxes(n,8)
            iou = polygon_box_iou(boxes[i], boxes, device=prediction.device) > iou_thres  # iou matrix
            weights = iou * x[:, 8][None]  # polygon box weights
            x[i, :8] = torch.mm(weights, x[:, :8]).float() / weights.sum(1, keepdim=True)  # merged polygon boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def polygon_nms_kernel(x, iou_thres):
    """
        non maximum suppression kernel for polygon-enabled boxes
        x is the prediction with boxes x[:, :8], confidence x[:, 8], class x[:, 9]
        Return the selected indices
    """

    unique_labels = x[:, 9].unique()
    _, scores_sort_index = torch.sort(x[:, 8], descending=True)
    x = x[scores_sort_index]
    x[:, :8] = order_corners(x[:, :8])
    indices = scores_sort_index
    selected_indices = []

    # Iterate through all predicted classes
    for unique_label in unique_labels:
        x_ = x[x[:, 9] == unique_label]
        indices_ = indices[x[:, 9] == unique_label]

        while x_.shape[0]:
            # Save the indice with the highest confidence
            selected_indices.append(indices_[0])
            if len(x_) == 1:
                break
            # Compute the IOUs for all other the polygon boxes
            iou = polygon_box_iou(x_[0:1, :8], x_[1:, :8], device=x.device, ordered=True).view(-1)
            # Remove overlapping detections with IoU >= NMS threshold
            x_ = x_[1:][iou < iou_thres]
            indices_ = indices_[1:][iou < iou_thres]

    return torch.LongTensor(selected_indices)


def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions:
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """

    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y)  # sort y
    x_sorted = torch.zeros_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        x_sorted[i] = x[i, y_indices[i]]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(y.shape[0]):
        y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()


def polygon_scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, 0::2] -= pad[0]  # x padding
    coords[:, 1::2] -= pad[1]  # y padding
    coords[:, :8] /= gain
    polygon_clip_coords(coords, img0_shape)  # inplace operation
    return coords


def polygon_clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0::2].clamp_(0, img_shape[1])  # x1x2x3x4
    boxes[:, 1::2].clamp_(0, img_shape[0])  # y1y2y3y4
