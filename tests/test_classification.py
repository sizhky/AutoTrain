from torch_snippets import P
import sys; sys.path.append(str(P('.').resolve()))
from main import train, validate

def no_test_classification_imagenette_training():
    train(task="classification", config="configs/classification_imagenette.ini")

def test_classification_imagenette_training():
    validate(task="classification", config="configs/classification_imagenette.ini", image='test_images/classification/imagenette_golf.jpeg')

