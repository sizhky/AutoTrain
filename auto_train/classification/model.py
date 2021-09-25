from torch_snippets import *
from torch_snippets.registry import Config, AttrDict, registry
from auto_train.classification.custom_functions import *
from auto_train.classification.timmy import create_timm_model

config = Config().from_disk(os.environ['CONFIG'])
config = AttrDict(registry.resolve(config))

from fastai.vision.all import *

def get_dataloaders(
    source,
    output,
    bs=64,
    item_tfms=[RandomResizedCrop(size=128, min_scale=0.35), FlipItem(0.5)],
    batch_tfms=RandomErasing(p=0.9, max_count=3)
):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(
            train_name=stem(config.training.data.train_dir),
            valid_name=stem(config.training.data.validation_dir)),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )
    dls = dblock.dataloaders(source=source, path=output, bs=bs, num_workers=8)
    return dls

dls = get_dataloaders(
    source=config.training.dir,
    output=config.training.scheme.output
)

model = create_timm_model(
    config.architecture.backbone.model,
    custom_head=config.architecture.head,
    n_out=config.project.num_classes)
    
learn = Learner(dls, model, splitter=default_split, metrics=[accuracy])
