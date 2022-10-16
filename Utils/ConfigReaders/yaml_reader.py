from Utils.ConfigReaders.base import BaseConfigReader
import yaml


class YamlConfigReader(BaseConfigReader):
    """yaml类型配置文件读取"""

    def __init__(self, yaml_path):
        BaseConfigReader.__init__(self)
        # open方法打开直接读出来
        rf = open(yaml_path, 'r', encoding='utf-8')
        self.src_info = yaml.load(rf.read())  # 用load方法转字典
        rf.close()
        self.info = self.delete()

    def configs_info(self):
        """返回读到的参数（字典）"""
        return self.info

    def delete(self):
        """删除无信息的键对值"""
        info = {}
        for k, v in self.src_info.items():
            if v is not None:
                info[k] = v
        return info
