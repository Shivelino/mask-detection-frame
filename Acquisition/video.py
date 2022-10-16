import os
import time

import cv2
from Acquisition.base import BaseAcquisition


class Video(BaseAcquisition):
    def __init__(self, **video_info):
        BaseAcquisition.__init__(self)
        self.video_info = video_info
        self.video_info_keys = video_info.keys()
        if "video_path" in self.video_info.keys():
            self.video_path = self.video_info["video_path"]
            self.video = cv2.VideoCapture(os.path.join(os.path.abspath(os.path.dirname(__file__)), self.video_path))
            self.video_display_fps = self.video_info["display_fps"]
        else:
            raise Exception("Video_path doesn't exist.")

    def get_frame(self, mode="rgb"):
        if mode is None:
            mode = self.video_info["img_mode"]
        _, frame = self.video.read()
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
            ret, frame = self.video.read()
            if ret:
                cv2.imshow("Video living, press q to ESC.", frame)
            else:
                break
            if cv2.waitKey(int(1000 / self.video_display_fps)) & 0xFF == ord('q'):
                break

    def __refresh_deque(self, com_deques_dict):
        while True:
            ret, frame = self.video.read()  # bgr图片
            if ret:
                print("acq refresh>>>>>>>>>")
                for _, com_deque in com_deques_dict.items():
                    com_deque.append(frame)
                time.sleep(1 / self.video_display_fps)
            else:
                break

    def run(self, com_deques_dict):
        self.__refresh_deque(com_deques_dict)


if __name__ == "__main__":
    info = {"video_path": r"E:\Study\_2022_fall\small_term\python\mask_detect\Threads\video_record\old.avi", "display_fps": 30}
    v = Video(**info)
    v.display()
    pass
