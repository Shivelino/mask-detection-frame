import cv2

from Acquisition.base import BaseAcquisition
from Exceptions.acq_exception import *


class Cam(BaseAcquisition):
    """相机抽象类"""

    def __init__(self):
        BaseAcquisition.__init__(self)
        self.cam = None
        self.cam_id = None

    def __connect_cam(self, *args, **kwargs):
        """内部函数。连接摄像头"""
        pass

    def __set_cam_params(self, *args, **kwargs):
        """内部函数。设置摄像头参数"""
        pass

    def __refresh_deque(self, *args, **kwargs):
        """内部函数。用于更新线程通信队列"""
        pass

    def run(self, *args, **kwargs):
        """线程执行函数。具体类的功能"""
        pass

    def get_frame(self, *args, **kwargs):
        """外部函数。获取一帧图像"""
        pass

    def get_config(self, *args, **kwargs):
        """获取内部配置参数"""
        pass

    def display(self, *args, **kwargs):
        """外部函数。测试用。实施摄像头画面"""
        pass

    def test(self, *args, **kwargs):
        """测试函数"""
        pass


class CamSingle(Cam):
    """单相机基础类"""

    def __init__(self, **cam_info):
        Cam.__init__(self)
        self.cam_num = 1
        self.cam_info = cam_info
        self.cam_info_keys = self.cam_info.keys()


class CamSingleHK(CamSingle):
    """海康单相机类"""

    def __init__(self, **cam_info):
        # cam_info = YamlConfigs(config_path)
        CamSingle.__init__(self, **cam_info)
        # cam初始化
        self.__connect_cam()
        self.__set_cam_params()

    def __connect_cam(self):
        if "rtsp" in self.cam_info_keys:
            # TODO: 如果rtsp一直读不到，就会死循环，错误处理没用。想办法解决
            self.cam_id = self.cam_info["rtsp"]
            self.cam = cv2.VideoCapture(self.cam_id)
        else:
            raise CamInfoIdNotFoundError("rtsp")

    def __set_cam_params(self):
        """
        注：海康相机不支持opencv改变参数。写在这儿是为了之后如果解决了这个问题可以使用。
        海康相机设置相关参数的方式是进入192.168.1.64的页面进行设置，设置之后摄像头会保存配置。
        """
        # 图片采集尺寸
        if "img_width" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_info["img_width"])
        if "img_height" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_info["img_height"])
        # 设置fps
        if "fps" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FPS, self.cam_info["fps"])
        # 亮度
        if "brightness" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_BRIGHTNESS, self.cam_info["brightness"])
        # 对比度
        if "contrast" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_CONTRAST, self.cam_info["contrast"])
        # 饱和度
        if "saturation" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_SATURATION, self.cam_info["saturation"])

    def __refresh_deque(self, com_deques_dict):
        while True:
            ret, frame = self.cam.read()  # bgr图片
            if ret:
                # print("acq refresh>>>>>>>>>")
                for _, com_deque in com_deques_dict.items():
                    com_deque.append(frame)

    def run(self, com_deques_dict):
        self.__refresh_deque(com_deques_dict)

    def get_frame(self, mode="bgr"):  # TODO: （*）不符合开闭原则
        if mode is None:
            mode = self.cam_info["img_mode"]
        _, frame = self.cam.read()
        if frame is None:
            raise AttributeError("Can't read images, most of time the reason is camera-offline.")
        if mode == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif mode == "bgr":
            return frame
        elif mode == "hsv":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError("Please input correct image mode! Your mode is {}, "
                             "not one of rgb, bgr or hsv.".format(mode))

    def get_config(self, *args, **kwargs):
        """获取内部配置参数"""
        pass

    def display(self, *args):
        while True:
            ret, frame = self.cam.read()
            if ret:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720))
                cv2.imshow("CamSingleHK living, press q to ESC.", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


class CamSingleUSB(CamSingle):
    """UBS单相机类"""

    def __init__(self, **cam_info):
        CamSingle.__init__(self, **cam_info)
        self.__connect_cam()
        self.__set_cam_params()

    def __connect_cam(self):
        if "id" in self.cam_info_keys:
            self.cam_id = self.cam_info["id"]
            self.cam = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        else:
            raise CamInfoIdNotFoundError("id")

    def __set_cam_params(self):
        # 图片采集尺寸
        if "img_width" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_info["img_width"])
        if "img_height" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_info["img_height"])
        # 设置fps
        if "fps" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_FPS, self.cam_info["fps"])
        # 亮度
        if "brightness" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_BRIGHTNESS, self.cam_info["brightness"])
        # 对比度
        if "contrast" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_CONTRAST, self.cam_info["contrast"])
        # 饱和度
        if "saturation" in self.cam_info_keys:
            self.cam.set(cv2.CAP_PROP_SATURATION, self.cam_info["saturation"])

    def get_frame(self, mode="rgb"):
        if mode is None:
            mode = self.cam_info["img_mode"]
        _, frame = self.cam.read()
        if frame is None:
            raise AttributeError("Can't read images, most of time the reason is camera-offline.")
        if mode == "rgb":
            return frame
        elif mode == "bgr":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif mode == "hsv":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError("Please input correct image mode! Your mode is {}, "
                             "not one of rgb, bgr or hsv.".format(mode))

    def display(self, *args):
        while True:
            ret, frame = self.cam.read()
            if ret:
                cv2.imshow("CamSingleUSB living, press q to ESC.", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __refresh_deque(self, com_deques_dict):
        while True:
            ret, frame = self.cam.read()  # bgr图片
            if ret:
                print("acq refresh>>>>>>>>>")
                for _, com_deque in com_deques_dict.items():
                    com_deque.append(frame)

    def run(self, com_deques_dict):
        self.__refresh_deque(com_deques_dict)


if __name__ == "__main__":
    pass
