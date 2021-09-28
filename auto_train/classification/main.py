from fastapi import APIRouter
import os; os.environ['CONFIG'] = 'configs/classification_imagenette.ini'

router = APIRouter()

@router.post('/train/')
async def train():
    from auto_train.classification.train import train_model
    train_model()

@router.post('/validate/')
async def validate():
    from auto_train.classification.infer import infer
    return infer()
