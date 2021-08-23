from icevision.all import *
from auto_train_object_detection.custom_functions import *
from torch_snippets.registry import Config, registry, AttrDict

config = Config().from_disk('config.ini')
config = AttrDict(registry.resolve(config))
data_dir = Path(config.training.dir)

images_dir = data_dir / 'images'
annotations_dir = data_dir / 'Annotations'
class_map = ClassMap(config.project.classes)
print(class_map)
parser = parsers.voc(annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map)
data_splitter = RandomSplitter((config.training.train_ratio, 1 - config.training.train_ratio))
train_records, valid_records = parser.parse(data_splitter)

presize = config.architecture.presize
size = config.architecture.size

train_tfms = config.training.preprocess
valid_tfms = config.testing.preprocess

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)


# model_type = models.torchvision.retinanet
assert config.architecture.model_type.count('.', 1), "Architecture should look like <base>.<model>"
a, b = config.architecture.model_type.split('.')
model_type = getattr(getattr(models, a), b)
backbone = getattr(model_type.backbones, config.architecture.backbone)(config.architecture.pretrained)
# backbone = model_type.backbones.resnet50_fpn(pretrained=True)
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map))

print(model)

train_dl = model_type.train_dl(train_ds, batch_size=4, num_workers=4, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=4, num_workers=4, shuffle=False)
model_type.show_batch(first(valid_dl), ncols=4)


metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
learn.lr_find()

learn.fine_tune(30, 1e-4, freeze_epochs=3)

from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
makedir('models')
save_torch_model_weights_from(model, 'models/0.pth')