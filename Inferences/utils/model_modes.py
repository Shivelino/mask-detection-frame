import os

from onnxruntime import InferenceSession
import numpy as np


class Model(object):
    """模型基类"""
    def __init__(self, *args, **kwargs):
        self.model = None

    def set(self, **kwargs):  # 这里设置成外部接口是为了方便测试
        """模型设置"""
        pass

    def infer(self, *args):  # 这里设置成外部接口是为了方便测试
        """模型推理"""
        pass


class ModelOnnx(Model):
    """Onnx模型"""
    def __init__(self, model_path):
        Model.__init__(self)
        self.model = InferenceSession(model_path)
        self.set()
        self.input_name = self.model.get_inputs()[0].name

    def set(self):
        pass

    def infer(self, inp):
        if isinstance(inp, np.ndarray):
            inp = np.array(inp, dtype=np.float32)
        return self.model.run([], {self.input_name: inp})


class ModelPth(Model):
    """Pytorch格式模型（pth，pt）"""
    pass


class ModelH5(Model):
    """H5模型"""
    pass


class ModelPb1(Model):
    """Tf1.x的pb模型"""
    pass


class ModelPb2(Model):
    """Tf2.x的pb模型"""
    pass
