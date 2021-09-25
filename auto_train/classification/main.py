from common.base import Task
from torch_snippets.registry import AttrDict, registry
from classification.custom_functions import *

class Classification(Task):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        self.config = AttrDict(registry.resolve(self.config))
        print(self.config)

def train_model(config):
    task = Classification(config)
    task.train()
