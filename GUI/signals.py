from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class BaseSignal(QObject):
    """基础信号类"""
    signal = pyqtSignal(int)

    def get_type(self, *args, **kwargs):
        """获取信号类型"""
        pass

    def send(self, *args, **kwargs):
        """发送信号"""
        pass


class SignalStr(BaseSignal):
    """Str刷新信号"""
    signal = pyqtSignal(str)

    def get_type(self):
        return str

    def send(self, text):
        self.signal.emit(text)


class SignalInt(BaseSignal):
    """Int刷新信号"""
    signal = pyqtSignal(int)

    def get_type(self):
        return int

    def send(self, num):
        self.signal.emit(num)


class SignalFloat(BaseSignal):
    """Float刷新信号"""
    signal = pyqtSignal(float)

    def get_type(self):
        return float

    def send(self, num):
        self.signal.emit(num)


class SignalList(BaseSignal):
    """List刷新信号"""
    signal = pyqtSignal(list)

    def get_type(self):
        return list

    def send(self, send_list):
        self.signal.emit(send_list)


class SignalDict(BaseSignal):
    """Dict刷新信号"""
    signal = pyqtSignal(dict)

    def get_type(self):
        return dict

    def send(self, send_dict):
        self.signal.emit(send_dict)


class SignalNdarray(BaseSignal):
    """ndarray刷新信号"""
    signal = pyqtSignal(np.ndarray)

    def get_type(self):
        return np.ndarray

    def send(self, ndarray):
        self.signal.emit(ndarray)


class Signals(object):
    @staticmethod
    def get_signal(signal_type):
        if signal_type == "int":
            return SignalInt()
        elif signal_type == "float":
            return SignalFloat()
        elif signal_type == "str":
            return SignalStr()
        elif signal_type == "list":
            return SignalList()
        elif signal_type == "dict":
            return SignalDict()
        elif signal_type == "ndarray":
            return SignalNdarray()
        else:
            raise Exception("Unsupported type.")  # TODO: 自定义错误


if __name__ == '__main__':
    test = Signals.get_signal("str")
    print(test.get_type())

    t = SignalNdarray()
    print(t.get_type())
