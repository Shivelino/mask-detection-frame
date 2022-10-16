"""
全局运行
"""
from Threads.child_threads import AcquisitionThread, ProcessThread, GuiThread
import time


if __name__ == "__main__":
    ta = AcquisitionThread()
    tp = ProcessThread()
    tg = GuiThread()
    ta.start()
    time.sleep(0.1)
    tp.start()
    time.sleep(0.1)
    tg.start()
