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
    