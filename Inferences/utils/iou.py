"""
iou计算函数
"""
import math

import shapely
import shapely.geometry
import shapely.geos
import torch

polygon_inter_union_cuda_enable = False
polygon_b_inter_union_cuda_enable = False


# ## 下面是车网络iou
def wh_iou(wh1, wh2, eps=1e-7):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# ## 下面是装甲板网络iou
def order_corners(boxes):
    """四点从左上顺时针排列"""
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


def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """

    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4, 2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)
    # 注掉原因是判断cuda.is_available()太耗时间
    # if torch.cuda.is_available() and polygon_inter_union_cuda_enable and boxes1.is_cuda:
    #     # using cuda extension to compute
    #     # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
    #     boxes1_ = boxes1.float().contiguous().view(-1)
    #     boxes2_ = boxes2.float().contiguous().view(-1)
    #     inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
    #
    #     inter_nan, union_nan = inter.isnan(), union.isnan()
    #     if inter_nan.any() or union_nan.any():
    #         inter2, union2 = polygon_inter_union_cuda(boxes1_,
    #                                                   boxes2_)  # Careful that order should be: boxes1_, boxes2_.
    #         inter2, union2 = inter2.T, union2.T
    #         inter = torch.where(inter_nan, inter2, inter)
    #         union = torch.where(union_nan, union2, union)
    # else:
    #     # using shapely (cpu) to compute
    inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0]  # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0]  # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0]  # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0]  # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                    w1, h1 = b1_x2[i] - b1_x1[i], b1_y2[i] - b1_y1[i] + eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU


def polygon_b_inter_union_cpu(boxes1, boxes2):
    """
        iou computation (polygon) with cpu for class Polygon_ComputeLoss in loss.py;
        Boxes and Anchors having the same shape: nx8;
        Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
    """

    n = boxes1.shape[0]
    inter = torch.zeros(n, )
    union = torch.zeros(n, )
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
        polygon2 = shapely.geometry.Polygon(boxes2[i, :].view(4, 2)).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter[i] = polygon1.intersection(polygon2).area
                union[i] = polygon1.union(polygon2).area
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_bbox_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
    """
        Compute iou of polygon boxes for class Polygon_ComputeLoss in loss.py via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

    if torch.cuda.is_available() and polygon_b_inter_union_cuda_enable and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside inter_union_cuda must be torch.cuda.float, not double type or half type
        boxes1_ = boxes1.float().contiguous().view(-1)
        boxes2_ = boxes2.float().contiguous().view(-1)
        inter, union = polygon_b_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.

        inter_nan, union_nan = inter.isnan(), union.isnan()
        if inter_nan.any() or union_nan.any():
            inter2, union2 = polygon_b_inter_union_cuda(boxes1_,
                                                        boxes2_)  # Careful that order should be: boxes1_, boxes2_.
            inter2, union2 = inter2.T, union2.T
            inter = torch.where(inter_nan, inter2, inter)
            union = torch.where(union_nan, union2, union)
    else:
        # using shapely (cpu) to compute
        inter, union = polygon_b_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0]  # n,
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0]  # n,
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0]  # n,
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0]  # n,
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                iou -= rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou.to(device) + (1 + eps))
                iou = iou.to(device) - (rho2.to(device) / c2.to(device) + v.to(device) * alpha.to(device))  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw.to(device) * ch.to(device) + eps  # convex area
            iou -= (c_area.to(device) - union.to(device)) / c_area.to(device)  # GIoU
    return iou  # IoU
