from icevision.all import *
from torch_snippets.registry import registry

if not hasattr(registry, 'augmentations'):
    registry.create('augmentations')

@registry.augmentations.register("get_train_transforms")
def get_train_transforms(size, presize):
    train_tfms = tfms.A.Adapter([
        tfms.A.Resize(size, size),
        # *tfms.A.aug_tfms(size=size, presize=presize),
        tfms.A.Normalize()
    ])
    return train_tfms

@registry.augmentations.register("get_val_transforms")
def get_val_transforms(size, presize):
    valid_tfms = tfms.A.Adapter([
        tfms.A.Resize(size, size),
        # *tfms.A.resize_and_pad(size=size),
        tfms.A.Normalize()
    ])
    return valid_tfms
