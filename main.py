from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
cli = Typer()


@cli.command()
def train(task=None, config=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from auto_train.classification.train import train_model
        train_model()


@cli.command()
def validate(task=None, folder=None, img_path=None, config=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from auto_train.classification.infer import infer
        output = infer(folder, img_path=img_path)
        pretty_json(output)


if __name__ == '__main__':
    cli()
