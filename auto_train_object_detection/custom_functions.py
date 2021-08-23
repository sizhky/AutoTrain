from icevision.all import *
from torch_snippets.registry import registry

if not hasattr(registry, 'augmentations'):
    registry.create('augmentations')

@registry.augmentations.register("get_train_transforms")
def get_train_transforms():
    train_tfms = tfms.A.Adapter([tfms.A.Normalize()])
    return train_tfms

@registry.augmentations.register("get_val_transforms")
def get_val_transforms():
    valid_tfms = tfms.A.Adapter([tfms.A.Normalize()])
    return valid_tfms
