import os
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QStyle

from GUI.interface import Ui_MainWindow
from GUI.ui_thread_classes import *
from GUI.signals import Signals
from Utils.ConfigReaders.yaml_reader import YamlConfigReader


class BaseUiBehavior(QMainWindow):
    """单个UI窗口行为基础类。在具体ui窗口行为中继承此类代替QMainWindow"""

    def __init__(self, *args, **kwargs):
        QMainWindow.__init__(self)

    def __ui_widgets_init(self, *args, **kwargs):
        """UI界面部件初始化"""
        pass

    def __ui_params_init(self, *args, **kwargs):
        """UI相关参数初始化"""
        pass

    def __ui_slots_init(self, *args, **kwargs):
        """UI插槽函数初始化"""
        pass

    def __ui_threads_init(self, *args, **kwargs):
        """UI子线程相关初始化"""
        pass

    def __exit(self, *args, **kwargs):
        """UI退出函数"""
        pass

    def run(self, *args, **kwargs):
        """UI执行函数"""
        pass


class UiBehaviorMain(BaseUiBehavior, Ui_MainWindow):
    """UI行为类（主界面）"""

    def __init__(self, signals, signal_deques):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        # super(self.__class__, self).__init__()  # 这句和之前两句效果一样但是更难理解
        self.configs = YamlConfigReader(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../Data/configs/gui.yaml")).configs_info()
        self.__ui_widgets_init()
        self.__ui_params_init()
        self.__ui_slots_init()
        self.__ui_threads_init(signals, signal_deques)

        # 直接开始跑了
        self.pushbutton_play_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pushbutton_play_pause.setText("Pause")
        self.src_display_thread()
        self.obd_display_thread()
        self.video_status = self.VIDEO_STATUSES[1]

    def __ui_widgets_init(self):
        # UI初始化
        self.setupUi(self)

    def __ui_params_init(self, *args, **kwargs):
        # 状态参数初始化
        self.VIDEO_STATUSES = (0, 1, 2)  # 0: init; 1: playing; 2: pause
        self.video_status = self.VIDEO_STATUSES[0]
        self.RECORD_STATUSES = (0, 1)  # 0: no; 1: yes
        self.record_status = self.RECORD_STATUSES[0]
        self.EXIT_STATUSES = (0, 1)  # 0: no; 1: yes
        self.exit_status = self.EXIT_STATUSES[0]
        # 特定self函数参数初始化
        self.video_writer = None  # record_button的写入类
        self.video_writer_counter = 0  # record_button的文件计数
        # 线程变量初始化
        self.src_display_timer = None  # src_display_thread线程

    def __ui_slots_init(self):
        # 时间触发(time trigger), 用于放映
        self.pushbutton_play_pause.clicked.connect(self.play_pause_button)
        # 录制触发(record trigger), 用于录制
        self.pushbutton_record.clicked.connect(self.record_button)
        # 退出触发(exit trigger), 用于退出
        self.pushbutton_exit.clicked.connect(self.exit_button)

    def __ui_threads_init(self, signals, signal_deques):
        # 这两个都是列表形式，按照下标index对齐。之后可以考虑全都使用字典。
        self.signals = signals
        self.signal_deques = signal_deques

    def __exit(self):
        if self.video_status == 1:
            self.src_display_timer.pause()
        print("\033[1;31mUI thread exit\033[0m")
        print("\033[1;31mPlease force to exit else threads\033[0m")
        self.thread().exit()

    def src_display_thread(self):
        """原视频放映线程"""
        self.src_display_timer = FixedFrequencyThread({"acq2gui_src": self.signals["acq2gui_src"]},
                                                      {"acq2gui_src": self.signal_deques["acq2gui_src"]},
                                                      frequency=self.configs["src_display_fps"])
        self.src_display_timer.start()
        self.src_display_timer.signal_class_dict["acq2gui_src"].signal[self.signals["acq2gui_src"].get_type()].connect(
            self.src_display_refresh)

    def src_display_refresh(self, img_bgr):
        """源视频线程刷新函数"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(img_rgb, (self.configs["src_display_width"], self.configs["src_display_height"]))

        temp_image = QImage(frame.flatten(), self.configs["src_display_width"], self.configs["src_display_height"],
                            QImage.Format_RGB888)
        temp_pixmap = QPixmap.fromImage(temp_image)
        self.video_window.setPixmap(temp_pixmap)
        # 保存视频
        if self.record_status == 1:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def obd_display_thread(self):
        """obd刷新线程"""
        self.obd_display_timer = FixedFrequencyThread({"inference2gui": self.signals["inference2gui"]},
                                                      {"inference2gui": self.signal_deques["inference2gui"]},
                                                      frequency=self.configs["obd_display__fps"])
        self.obd_display_timer.start()
        self.obd_display_timer.signal_class_dict["inference2gui"].signal[
            self.signals["inference2gui"].get_type()].connect(self.obd_display_refresh)

    def obd_display_refresh(self, img_bgr):
        """obd刷新函数"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(img_rgb, (self.configs["obd_display_width"], self.configs["obd_display_height"]))

        temp_image = QImage(frame.flatten(), self.configs["obd_display_width"], self.configs["obd_display_height"],
                            QImage.Format_RGB888)
        temp_pixmap = QPixmap.fromImage(temp_image)
        self.obd_window.setPixmap(temp_pixmap)

    def play_pause_button(self):
        """放映暂停按钮"""
        if self.video_status == 0:
            self.pushbutton_play_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.pushbutton_play_pause.setText("Pause")
            # self.src_display_thread()
            # self.obd_display_thread()
            self.video_status = self.VIDEO_STATUSES[1]
        elif self.video_status == 1:
            self.pushbutton_play_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.pushbutton_play_pause.setText("Play")
            self.src_display_timer.pause()
            self.obd_display_timer.pause()
            # self.cam.cam.data_stream[0].flush_queue()  # del cache
            self.video_status = self.VIDEO_STATUSES[2]
        elif self.video_status == 2:
            self.pushbutton_play_pause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.pushbutton_play_pause.setText("Pause")
            self.src_display_timer.start()
            self.obd_display_timer.start()
            self.video_status = self.VIDEO_STATUSES[1]

    def record_button(self):
        """录制按钮"""
        if self.record_status == 0 and self.video_status == 1:
            self.pushbutton_record.setText("Recording...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            if os.path.exists("./video_record") is False:
                os.mkdir("./video_record")
            self.video_writer = cv2.VideoWriter("./video_record/{}.avi".format(str(self.video_writer_counter)),
                                                fourcc, 20.0,
                                                (self.configs["src_display_width"], self.configs["src_display_height"]))
            self.record_status = self.RECORD_STATUSES[1]
        elif (self.record_status == 1 and self.video_status == 1) or \
                (self.record_status == 1 and self.video_status == 2):
            self.pushbutton_record.setText("Record")
            self.video_writer.release()
            self.video_writer = None
            self.video_writer_counter += 1
            self.record_status = self.RECORD_STATUSES[0]

    def exit_button(self):
        """退出按钮"""
        self.__exit()


class UiMainThread(object):
    """UI主线程。用于开启某个具体的ui界面作为ui主线程"""

    def __init__(self):
        pass

    def __app_init(self):
        """QApplication初始化"""
        self.app = QApplication(sys.argv)

    def run(self, com_deques_dict):
        """起始函数"""
        self.__app_init()
        ui_main = UiBehaviorMain({"acq2gui_src": Signals.get_signal("ndarray"),
                                  "inference2gui": Signals.get_signal("ndarray")
                                  # TODO: 这里得根据输进来的值进行自动化给值，不能直接写死或者把signal初始化丢给Threads
                                  },
                                 # [SignalNdarray, SignalNdarray, SignalStr],
                                 com_deques_dict)  # NOTES 改队列数量记得改左边sig数量
        ui_main.show()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    pass
