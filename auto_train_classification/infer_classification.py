import timm
import torch
import torch.nn as nn
import os
from pathlib import Path as P

from PIL import Image
from torch_snippets import parent
import sys; sys.path.append(str(parent(parent(__file__)).resolve()))

from torch_snippets import *
from torch_snippets.registry import *
from auto_train_classification.custom_functions import *

settings = AttrDict(Config().from_disk(os.environ['CONFIG']))
settings = AttrDict(registry.resolve(settings))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = settings.architecture.image_size
image_transforms = settings.training.preprocess

image_cache_dir = os.path.join(os.path.dirname(__file__), 'image-cache')
os.makedirs(image_cache_dir, exist_ok=True)

def get_transformed_image(filepath):
    with open(filepath, mode='rb') as f:
        image = Image.open(f).convert('RGB')
    return image_transforms(image).to(device)

class ImageClassifier(object):
    def __init__(self):
        # self.model = models.resnet18(pretrained=True)
        self.model = timm.create_model(
            settings.architecture.backbone.model, 
            pretrained=True)
        self.model.classifier = nn.Linear(
            settings.architecture.backbone.vector_size,
            settings.project.num_classes
        )
        self.model = self.model.to(device)
        folder = Glob(settings.training.scheme.output)[0]
        print(folder)
        self.load(folder/'model_best.pth.tar')

    def load(self, path):
        state = torch.load(path)
        if 'state_dict' in state:
          state = state['state_dict']
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image_paths):
        images = torch.stack([get_transformed_image(path) for path in image_paths])
        with torch.no_grad():
            preds = self.model(images).cpu().data
            preds = torch.nn.functional.softmax(preds,-1)
        confs, classes = preds.max(-1)
        preds = pd.DataFrame({
            'confidence': confs, 
            'prediction': classes,
            'path': image_paths,
            'image_id': [fname(f) for f in image_paths]
        })
        return preds

import typer

app = typer.Typer()

@app.command()
def predict(folder:P):
    model = ImageClassifier()
    files = [str(f) for f in folder.glob('*.jpg')]
    return model.predict(files)

if __name__ == '__main__':
    app()
    