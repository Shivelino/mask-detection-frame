import itertools

import cv2
import torch

from Inferences.utils.model_modes import *
from Inferences.utils.nms import polygon_non_max_suppression, polygon_scale_coords, \
    non_max_suppression, scale_coords
from Utils.ConfigReaders.yaml_reader import YamlConfigReader


class InferStructure(object):
    """推理结构抽象类"""

    def __init__(self):
        pass

    def __pre_process(self, *args):
        """预处理"""
        pass

    def __inference(self, *args):
        """网络推理"""
        pass

    def __post_process(self, *args):
        """后处理"""
        pass

    def process(self, *args):
        """infer处理接口"""
        pass


class ObDInfer(InferStructure):
    """目标检测领域推理基础类"""

    def __init__(self):
        InferStructure.__init__(self)

    def nms(self):
        """目标检测nms计算。如果有已经造好的轮子就从外部引用，不用自己写了"""
        pass


class YoloV5RectInferOnnx(ObDInfer, ModelOnnx):
    """YoloV5矩形框Onnx模型推理"""

    def __init__(self):
        self.configs = YamlConfigReader(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../Data/configs/mask_net.yaml")).configs_info()
        # 父类初始化
        ObDInfer.__init__(self)
        ModelOnnx.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), self.configs["model_path"]))
        # anchor初始化
        self.anchors = torch.stack(
            [torch.tensor(self.configs["anchors"][i], dtype=torch.float32) / 2 / self.configs["strides"][i] for i in
             range(self.configs["nl"])]).to(self.configs["device"])

    def __pre_process(self, img_bgr):
        self.img_shape = np.shape(img_bgr)[0:2]
        assert isinstance(img_bgr, np.ndarray), "Pic must be ndarray."
        img_shape = np.shape(img_bgr)
        if img_shape[0:2] != (self.configs["img_size"][0], self.configs["img_size"][1]):  # 网络输入size1024
            img_bgr = cv2.resize(img_bgr, (self.configs["img_size"][0], self.configs["img_size"][1]))
        inp = np.array(torch.tensor(img_bgr / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(dim=0))
        return inp  # Onnx需要输入ndarray

    def __inference(self, inp):
        return self.infer(inp)

    def __post_process(self, prediction):
        prediction = [torch.tensor(x) for x in prediction]
        pred = self.__layer_fusion(prediction)[0]
        pred = non_max_suppression(pred, conf_thres=self.configs["nms__conf_threshold"],
                                   iou_thres=self.configs["nms__iou_threshold"])
        det = pred[0]  # infer肯定只有一张图片
        results = []
        if len(det):
            det[:, :4] = scale_coords(self.configs["img_size"], det[:, :4],
                                      (self.img_shape[0], self.img_shape[1])).round()
            for *xyxy, conf, cls in reversed(det):
                pos = torch.stack(xyxy, dim=0).view(-1, 2)
                results.append((pos.to("cpu").detach().numpy().astype(np.int32),
                                conf.to("cpu").item(), cls.to("cpu").item()))
        return results

    def process(self, img_bgr):
        inp = self.__pre_process(img_bgr)
        prediction = self.__inference(inp)
        return self.__post_process(prediction)

    def __layer_fusion(self, x):  # x就是特征层列表
        """特征层融合"""
        z = []
        for i in range(self.configs["nl"]):
            # 已经是x(bs,3,20,20,6)
            bs, na, ny, nx, no = x[i].shape
            grid, anchor_grid = self._make_grid(na=na, nx=nx, ny=ny, i=i)

            y = x[i].sigmoid()
            xy = (y[..., 0:2] * 2 - 0.5 + grid) * self.configs["strides"][i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, no))
        return torch.cat(z, 1), x

    def _make_grid(self, na=3, nx=20, ny=20, i=0):
        """生成每个grid_box的数据对应的grid和anchor，用于定位"""
        yv, xv = torch.meshgrid(
            [torch.arange(ny, device=self.configs["device"]), torch.arange(nx, device=self.configs["device"])])
        grid = torch.stack((xv, yv), 2).expand((1, na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * torch.tensor(self.configs["strides"][i]).to(self.configs["device"])) \
            .view((1, na, 1, 1, 2)).expand((1, na, ny, nx, 2)).float()
        return grid, anchor_grid


class NetProcess(object):
    def __init__(self):
        self.car_infer = YoloV5RectInferOnnx()

    def run(self, com_deques_dict):
        while True:
            processed_img = self.net_process(com_deques_dict["acq2inference_net"])
            if processed_img is not None:
                com_deques_dict["inference2gui"].append(processed_img)

    def net_process(self, com_deque):
        """网络处理"""
        if len(com_deque) == 0:
            return None
        img_bgr = com_deque.popleft()
        img2draw = img_bgr.copy()
        print("infer processing.......")
        car_boxes = self.car_infer.process(img_bgr)
        for res in car_boxes:
            if int(res[2]) == 0:
                cv2.rectangle(img2draw, (res[0][0][0] - 50, res[0][0][1] - 50), (res[0][1][0] + 50, res[0][1][1] + 50),
                              (0, 0, 255), thickness=4)
                cv2.putText(img2draw, "NO MASK", (res[0][0][0] - 50, res[0][0][1] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)
            elif int(res[2]) == 1:
                cv2.rectangle(img2draw, (res[0][0][0] - 50, res[0][0][1] - 50), (res[0][1][0] + 50, res[0][1][1] + 50),
                              (0, 255, 0), thickness=4)
                cv2.putText(img2draw, "MASKED", (res[0][0][0] - 50, res[0][0][1] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                pass
        return img2draw


if __name__ == "__main__":
    tt = YoloV5RectInferOnnx()
    img = cv2.imread(r"E:\Study\_2022_fall\small_term\python\dataset_mask\images\0.jpg")
    print(tt.process(img))
    # tt = YoloV5LayeredPolygonInferOnnx()
    # img = cv2.imread(r"F:\hanjia_2022_work\rm2022\LayeredMultiNet\utils\2900.jpg")
    # print(tt.process(img))
