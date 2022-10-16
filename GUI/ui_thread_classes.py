import time
from PyQt5.QtCore import QThread, QMutex, QMutexLocker
from GUI.signals import SignalList


class BaseQthread(QThread):
    """用于在UI主线程开ui子线程"""

    def __init__(self):
        QThread.__init__(self)

    def run(self, *args, **kwargs):
        """线程执行函数"""
        pass

    def pause(self, *args, **kwargs):
        """线程暂停"""
        pass

    def is_paused(self, *args, **kwargs):
        """获取线程是否暂停"""
        pass


class TimerThread(BaseQthread):
    """定频计时触发子线程。串口发送和页面刷新都需要这个新建对应的TimerThread"""

    def __init__(self, signal_class, signal_deque, frequency=1):
        """
        :param signal_class: 信号类
        :param signal_deque: 信号通信队列
        :param frequency: 频率
        """
        BaseQthread.__init__(self)
        self.stopped = False
        self.frequency = frequency
        self.timeSignal = signal_class
        self.signal_deque = signal_deque
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            if len(self.signal_deque) == 0:  # 队列为空时的处理
                time.sleep(1 / 1000)  # 休眠一个极小量防止抢占太多资源
                continue
            self.timeSignal.send(self.signal_deque.popleft())
            # print("test")
            time.sleep(1 / self.frequency)

    def pause(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_paused(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequency = fps


class LogThread(BaseQthread):
    """日志线程。这个之后也可修改为接收多个队列然后返回列表形式的线程"""

    def __init__(self, signal_deques, frequency=100):
        BaseQthread.__init__(self)
        self.stopped = False
        self.timeSignal = SignalList()  # TODO：这里不能直接调用signal
        self.signal_deques = signal_deques
        self.mutex = QMutex()
        self.frequency = frequency

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            send_list = []
            for signal_deque in self.signal_deques:
                if len(signal_deque) > 0:
                    send_list.append(signal_deque.popleft())
            # self.timeSignal.send(send_list)
            self.timeSignal.send(["working1111111", "working2222222"])  # TODO: 测试
            time.sleep(1 / self.frequency)

    def pause(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_paused(self):
        with QMutexLocker(self.mutex):
            return self.stopped


class FixedFrequencyThread(BaseQthread):
    """定频计时触发子线程"""
    def __init__(self, signal_class_dict, signal_deque_dict, frequency):
        """
        :param signal_class_dict: 信号类  ep: {"name": SignalInt()}
        :param signal_deque_dict: 信号通信队列  ep: {"name": com_deque}
        :param frequency: 频率

        Note:
            1, signal_class_dict和signal_deque_dict的键名必须一一对应
        """
        BaseQthread.__init__(self)
        self.stopped = False
        assert signal_class_dict.keys() == signal_deque_dict.keys(), "Key names must correspond one by one."
        self.keys = signal_class_dict.keys()
        self.signal_class_dict = signal_class_dict
        self.signal_deque_dict = signal_deque_dict
        self.frequency = frequency
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            # FIXME: 字典遍历assert信号type
            for name in self.signal_deque_dict.keys():
                if len(self.signal_deque_dict[name]) == 0:  # 队列为空时的处理
                    time.sleep(1 / 1000)  # 休眠一个极小量防止抢占太多资源
                    continue
                self.signal_class_dict[name].send(self.signal_deque_dict[name].popleft())
            time.sleep(1 / self.frequency)

    def pause(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_paused(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def __set_fps(self, fps):
        """设置fps（测试）"""
        self.frequency = fps


if __name__ == '__main__':
    FixedFrequencyThread({"name1": None, "name2": None}, {"name2": None, "name1": None}, 1)
