class CamInfoIdNotFoundError(Exception):
    """无相机标识错误"""
    def __init__(self, cam_id_mode):
        Exception.__init__(self, "Kwargs don't have {} to find a cam.".format(cam_id_mode).upper())


class CamOpenFailError(Exception):
    """打开相机失败错误"""
    def __init__(self, cam_id):
        Exception.__init__(self, "Open cam failed, most of the time cam is offline. Cam_id:{}.".format(cam_id))


class CamStreamError(Exception):
    """相机流错误基础类"""
    def __init__(self, description):
        Exception.__init__(self, description)


class CamStreamOnError(CamStreamError):
    """相机流开启错误"""
    def __init__(self, cam_id):
        super(CamStreamOnError, self).__init__("Error when OPEN cam stream. Cam_id:{}.".format(cam_id))


class CamStreamOffError(CamStreamError):
    """相机流关闭错误"""
    def __init__(self, cam_id):
        super(CamStreamOffError, self).__init__("Error when CLOSE cam stream. Cam_id:{}.".format(cam_id))


if __name__ == "__main__":
    raise CamInfoIdNotFoundError("sde")
