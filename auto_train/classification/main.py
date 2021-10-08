from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from io import BytesIO
from PIL import Image
import os
from loguru import logger

router = APIRouter()

@router.post('/train/')
async def train(background_tasks: BackgroundTasks):
    from auto_train.classification.train import train_model
    # background_tasks.add_task(train_model, 'configs/classification_imagenette.ini')
    return {
        'message': 'Model training has started',
    }

@router.post('/validate/')
async def validate(img: UploadFile = File(...)):
    from auto_train.classification.infer import infer
    image = Image.open(BytesIO(img.file.read()))
    logger.info(img.filename)
    img_path = f'test_images/from_api/{img.filename}'
    image.save(img_path)
    return infer(img_path)
