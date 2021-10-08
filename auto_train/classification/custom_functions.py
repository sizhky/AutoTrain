from torch_snippets.registry import registry
from torchvision import transforms
from torch import nn

if not hasattr(registry, 'preprocess_function'):
    registry.create('preprocess_function')

@registry.preprocess_function.register("my_preprocess")
class Preprocess:
    def __init__(self, image_size):
        self.image_size = image_size
        self.image_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        return self.image_transforms(image)

        
if not hasattr(registry, 'head'):
    registry.create('head')
    
@registry.head.register("custom_head")
class custom_head(nn.Sequential):
    def __init__(self, head_input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(head_input_size, head_input_size//2),
            nn.ReLU(inplace=True),
            nn.Linear(head_input_size//2, num_classes),
        )
        
    def forward(self, x):
        return self.model(x)
    
if not hasattr(registry, 'postprocess_function'):
    registry.create('postprocess_function')

@registry.postprocess_function.register('convert_imagenette_label_mapping')
def post_process_function():
    def imagenette_label_mapping(pred):
        label_mapping = dict(
            n01440764='tench',
            n02102040='English springer',
            n02979186='cassette player',
            n03000684='chain saw',
            n03028079='church',
            n03394916='French horn',
            n03417042='garbage truck',
            n03425413='gas pump',
            n03445777='golf ball',
            n03888257='parachute'
        )
        pred['prediction'] = label_mapping[pred['prediction']]
        return pred
    return imagenette_label_mapping


if not hasattr(registry, 'load_function'):
    registry.create('load_function')

def load_imagenette():
    def load_data_bunch(config):
        from torch_snippets import stem
        from fastai.vision.all import (
            DataBlock, ImageBlock, CategoryBlock,
            GrandparentSplitter, get_image_files,
            parent_label,
            RandomResizedCrop, FlipItem,
            RandomErasing
        )
        bs = config.training.scheme.batch_size
        item_tfms=[
            RandomResizedCrop(size=128, min_scale=0.35),
            FlipItem(0.5)],
        batch_tfms=RandomErasing(p=0.9, max_count=3)
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
        dls = dblock.dataloaders(
            source=config.training.dir, path=config.training.scheme.output,
            bs=bs, num_workers=8
        )
        return dls
    return load_data_bunch

@registry.load_function.register('load_rooftops')
def load_rooftops():
    from fastai.vision.all import (
        DataBlock, ImageBlock, CategoryBlock, 
        get_image_files
    )
    from torch_snippets import pd, P
    def load_data_bunch(config):
        from fastai.vision.all import (
            DataBlock, ImageBlock, CategoryBlock,
            GrandparentSplitter, get_image_files,
            parent_label
        )
        labels = pd.read_csv(P(config.training.dir)/'labels.csv', header=None)
        labels.columns = 'fname,class'.split(',')
        labels.set_index('fname', inplace=True)
        labels = labels.to_dict()['class']
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=lambda x: labels.get(P(x).stem),
        )

        dls = dblock.dataloaders(
            source=config.training.dir, path='./',
            bs=1, num_workers=8
        )
        return dls
    return load_data_bunch



