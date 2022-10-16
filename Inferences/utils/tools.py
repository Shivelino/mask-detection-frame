"""
其他常用函数
"""
import numpy as np
import torch


# ## 下面是车网络tools
def xyxy2xywh(xyxy):
    """bbox标注从xmin-ymin-xmax-ymax到x_center-y_center-width-height"""
    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    return np.array([xyxy[0] + w / 2, xyxy[1] + h / 2, w, h])


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
