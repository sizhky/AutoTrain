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
            self.dls = self.get_dataloaders(
                source=config.training.dir,
                output=config.training.scheme.output)
            self.learn = Learner(
                self.dls, self.model,
                splitter=default_split,
                metrics=[accuracy])

    def get_dataloaders(
        self, source, output, bs=64,
        item_tfms=[
            RandomResizedCrop(size=128, min_scale=0.35),
            FlipItem(0.5)],
        batch_tfms=RandomErasing(p=0.9, max_count=3)
    ):
        training_dir = str(P(self.config.training.dir).resolve())
        if not os.path.exists(training_dir):
            print(f'downloading data to {training_dir}...')
            self.download_data()

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            splitter=GrandparentSplitter(
                train_name=stem(self.config.training.data.train_dir),
                valid_name=stem(self.config.training.data.validation_dir)),
            get_items=get_image_files,
            get_y=parent_label,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms
        )
        dls = dblock.dataloaders(
            source=source, path=output,
            bs=bs, num_workers=8
        )
        return dls

