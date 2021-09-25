from torch_snippets.registry import *
from torch_snippets.markup import *

import os
c = Config().from_disk(os.environ['CONFIG'])
c = AttrDict(c)

content = {
    'model': c.architecture.backbone.model,
    'data_dir': c.training.dir,
    'num_classes': c.architecture.head.num_classes,
    'epochs': c.training.scheme.epochs,
    'mixup': c.training.scheme.mixup,
    'cutmix': c.training.scheme.cutmix,
    'mixup_switch_prob': c.training.scheme.mixup_switch_prob,
    'amp': c.training.scheme.amp,
    'aa': c.training.scheme.aa,
    'output': c.training.scheme.output,
    'batch_size': c.training.scheme.batch_size
}

if c.training.scheme.initial_checkpoint:
    content['initial_checkpoint'] =  c.training.scheme.initial_checkpoint

from torch_snippets.markup import pretty_json
pretty_json(content)

write_yaml(content, 'auto_train_classification/config.yml')