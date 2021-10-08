from torch_snippets import stem, logger, os, unzip_file, P
from torch_snippets.registry import Config, AttrDict, registry
from fastai.vision.all import *
import torch_snippets
from auto_train.classification.custom_functions import *
from auto_train.classification.timmy import create_timm_model

from auto_train.common import Task

class ClassificationModel(Task):
    def __init__(self, config, inference_only=True):
        super().__init__(config)
        config = self.config

        self.model = create_timm_model(
            config.architecture.backbone.model,
            custom_head=config.architecture.head,
            n_out=config.project.num_classes)
        if inference_only:
            self.dls = self.get_dataloaders()
            self.learn = Learner(
                self.dls, self.model,
                splitter=default_split,
                metrics=[accuracy])

    def get_dataloaders(self):
        training_dir = str(P(self.config.training.dir).expanduser())
        if not os.path.exists(training_dir):
            print(f'downloading data to {training_dir}...')
            self.download_data()
        
        dls = self.config.training.data.load_datablocks(self.config)
        return dls

