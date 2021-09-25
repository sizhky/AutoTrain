from icevision.all import *
from torch_snippets import Glob, parent, sys, P
sys.path.append(str(P().resolve()))
from auto_train_object_detection.preprocess import *
from torch_snippets.registry import Config, registry, AttrDict

config = Config().from_disk(os.environ['CONFIG'])
config = AttrDict(registry.resolve(config))
data_dir = Path(config.training.dir)

images_dir = config.training.images_dir
annotations_file = config.training.annotations_file
print(f'{len(Glob(images_dir))} images found')
class_map = ClassMap(config.project.classes)
parser = parsers.coco(annotations_file=annotations_file, img_dir=images_dir)
data_splitter = RandomSplitter((config.training.train_ratio, 1 - config.training.train_ratio))
logger.info(f'\nCLASSES INFERRED FROM {config.training.annotations_file}: {parser.class_map}')
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