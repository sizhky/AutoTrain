import os
from torch_snippets.registry import Config

class Task:
    def __init__(self, config):
        self.get_config(config)

    def get_config(self, config):
        if config is None:
            config = os.environ['CONFIG']
        self.config = Config().from_disk(config)
       
