from icevision.all import *
from torch_snippets import Glob, parent, sys, P
from torch_snippets.registry import Config, registry, AttrDict

from auto_train.common import Task
from auto_train.object_detection.preprocess import *

class ObjectDetection(Task):
    def __init__(self, config, inference_only=False):
        super().__init__(config)
        config = self.config

        self.create_parser()
        self.model = self.create_model()

        if not inference_only:
            self.dls = self.get_dataloaders()
            self.learn = self.model_type.fastai.learner(
                dls=self.dls, 
                model=self.model, 
                metrics=[COCOMetric(metric_type=COCOMetricType.bbox)])

    def get_dataloaders(self):
        config = self.config
        data_splitter = RandomSplitter((config.training.train_ratio, 1 - config.training.train_ratio))
        train_records, valid_records = self.parser.parse(data_splitter)

        train_tfms = config.training.preprocess
        valid_tfms = config.testing.preprocess

        train_ds = Dataset(train_records, train_tfms)
        valid_ds = Dataset(valid_records, valid_tfms)

        train_dl = self.model_type.train_dl(train_ds, batch_size=self.config.training.scheme.batch_size, num_workers=4, shuffle=True)
        valid_dl = self.model_type.valid_dl(valid_ds, batch_size=self.config.training.scheme.batch_size, num_workers=4, shuffle=False)
        return train_dl, valid_dl

    def create_model(self):
        config = self.config
        extra_args = config.architecture.extra_args.to_dict()
        assert config.architecture.model_type.count('.', 1), "Architecture should look like <base>.<model>"
        a, b = config.architecture.model_type.split('.')
        self.model_type = getattr(getattr(models, a), b)
        backbone = getattr(
                self.model_type.backbones,
                config.architecture.backbone)(config.architecture.pretrained)
        model = self.model_type.model(
            backbone=backbone(pretrained=True), 
            num_classes=len(self.parser.class_map),
            **extra_args
        )
        return model

    def create_parser(self):
        config = self.config
        training_dir = str(P(config.training.dir).expanduser())
        if not os.path.exists(training_dir):
            self.download_data()

        annotations_file = config.training.annotations_file
        images_dir = config.training.images_dir
        self.parser = parsers.coco(annotations_file=annotations_file, img_dir=images_dir)
        self.class_map = self.parser.class_map
        logger.info(f'\nCLASSES INFERRED FROM {config.training.annotations_file}: {self.parser.class_map}')

