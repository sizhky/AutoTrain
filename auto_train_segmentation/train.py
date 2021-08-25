from icevision.all import *
from auto_train_object_detection.custom_functions import *
from torch_snippets.markup import read_json, write_json
from torch_snippets.registry import Config, registry, AttrDict
from torch_snippets import Glob, parent

config = Config().from_disk('config.ini')
config = AttrDict(registry.resolve(config))

from torch_snippets import *
from icevision.all import *

def incby1(d):
    for k,v in d.items():
        if k in ['id', 'image_id', 'category_id']:
            d[k] = v+1
        if isinstance(v, list):
            [incby1(i) for i in v if isinstance(i, dict)]
        if isinstance(v, dict):
            incby1(v)

x = read_json(config.training.annotations_file)
incby1(x)
write_json(x, '/tmp/intermediate-file.json')

parser = parsers.COCOMaskParser(
    '/tmp/intermediate-file.json', 
    config.training.images_dir
)

data_splitter = RandomSplitter((config.training.train_ratio, 1 - config.training.train_ratio))
train_records, valid_records = parser.parse(data_splitter)

presize = config.architecture.presize
size = config.architecture.size

train_tfms = config.training.preprocess
valid_tfms = config.testing.preprocess

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)


# model_type = models.torchvision.retinanet
extra_args = config.architecture.extra_args
assert config.architecture.model_type.count('.', 1), "Architecture should look like <base>.<model>"
a, b = config.architecture.model_type.split('.')
model_type = getattr(getattr(models, a), b)
backbone = getattr(model_type.backbones, config.architecture.backbone)(config.architecture.pretrained)
# backbone = model_type.backbones.resnet50_fpn(pretrained=True)
model = model_type.model(
    backbone=backbone(pretrained=True), 
    num_classes=len(parser.class_map),
    **extra_args
)

train_dl = model_type.train_dl(train_ds, batch_size=4, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=4, num_workers=4, shuffle=False)
# model_type.show_batch(first(valid_dl), ncols=4)


metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
# learn.lr_find()

training_scheme = config.training.scheme

with learn.no_bar():
    learn.fine_tune(
        training_scheme.epochs, 
        training_scheme.lr, 
        freeze_epochs=training_scheme.freeze_epochs
    )

from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
makedir(parent(training_scheme.output_path))
save_torch_model_weights_from(model, training_scheme.output_path)