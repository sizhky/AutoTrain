from icevision.all import *
from torch_snippets import Glob, parent, sys, P
sys.path.append(str(P().resolve()))
from auto_train_object_detection.custom_functions import *
from torch_snippets.registry import Config, registry, AttrDict

config = Config().from_disk('config.ini')
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
# learn.lr_find()
import typer
app = typer.Typer()

@app.command()
def find_best_learning_rate():
    suggested_lrs = learn.lr_find(show_plot=False)
    recorder = learn.recorder
    skip_end = 5
    lrs    = recorder.lrs    if skip_end==0 else recorder.lrs   [:-skip_end]
    losses = recorder.losses if skip_end==0 else recorder.losses[:-skip_end]
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    fig.savefig(f'{config.project.location}/find_lr_plot.png')
    logger.info(f'LR Plot is saved at {config.project.location}/find_lr_plot.png')
    logger.info(f'Suggested LRs: {suggested_lrs.lr_min} and {suggested_lrs.lr_steep}')
    
@app.command()
def train_model(lr:float=None):
    from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
    try:
        load_torch_model_weights_to(model, config.training.scheme.resume_training_from)
    except:
        logger.info('Training from scratch!')
        
    training_scheme = config.training.scheme

    lr = lr if lr is not None else training_scheme.lr
    with learn.no_bar():
        learn.fine_tune(
            training_scheme.epochs, 
            lr, 
            freeze_epochs=training_scheme.freeze_epochs
        )

    makedir(parent(training_scheme.output_path))
    save_torch_model_weights_from(model, training_scheme.output_path)
    
if __name__ == '__main__':
    app()