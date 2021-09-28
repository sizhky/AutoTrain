from fastapi import FastAPI
from typer import Typer
from torch_snippets.markup import pretty_json

from auto_train.classification.main import router as classification_router

ALLOWED_TASKS = [
    'classification',
    'object_detection',
    'segmentation'
]

app = FastAPI()
app.include_router(classification_router, prefix='/classification')

cli = Typer()

@cli.command()
def train(task=None, config=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from auto_train.classification.train import train_model
        train_model()

@cli.command()
def infer(task=None, folder=None, config=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from auto_train.classification.infer import infer
        output = infer(folder)
        pretty_json(output)

if __name__ == '__main__':
    cli()

