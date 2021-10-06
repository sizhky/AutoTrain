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
