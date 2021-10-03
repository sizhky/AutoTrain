from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer
from torch_snippets.markup import pretty_json



from auto_train.classification.main import router as classification_router
from auto_train.object_detection.main import router as object_detection_router
from auto_train.object_detection.main import router as segmentation_router

ALLOWED_TASKS = [
    'classification',
    'object_detection',
    'segmentation'
]

app = FastAPI()
app.include_router(classification_router, prefix='/classification')
app.include_router(object_detection_router, prefix='/object_detection')
app.include_router(segmentation_router, prefix='/segmentation')

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
        from auto_train.classification.train import train_model as train_classifier
        train_classifier(config)

    if task == 'object_detection':
        from auto_train.object_detection.train import train_model as train_object_detector
        train_object_detector(config)

    if task == 'segmentation':
        from auto_train.segmentation.train import train_model as train_segmentation
        train_segmentation(config)


@cli.command()
def validate(task=None, config=None, image=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from auto_train.classification.infer import infer as classification_infer
        output = classification_infer(image)
        print(output)

    if task == 'object_detection':
        from auto_train.object_detection.infer import infer as object_detection_infer
        output = object_detection_infer(image)
        print(output)

    if task == 'segmentation':
        from auto_train.segmentation.infer import infer as segmentation_infer
        output = segmentation_infer(image)
        print(output)

if __name__ == '__main__':
    cli()
