from torch_snippets import rand
from fastapi import APIRouter, BackgroundTasks
import os; os.environ['CONFIG'] = 'configs/classification_imagenette.ini'

router = APIRouter()

@router.post('/train/')
async def train(background_tasks: BackgroundTasks):
    from auto_train.classification.train import train_model
    train_id = rand()
    train_model(train_id)
    return {
        'message': 'Model training has started',
        'train_id': train_id
    }

@router.post('/validate/')
async def validate(background_tasks: BackgroundTasks):
    from auto_train.classification.infer import infer
    return infer()
