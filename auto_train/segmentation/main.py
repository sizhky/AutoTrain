from torch_snippets import rand
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from io import BytesIO
from PIL import Image
import os
from loguru import logger

router = APIRouter()

@router.post('/train/')
async def train(background_tasks: BackgroundTasks):
    from auto_train.segmentation.train import train_model
    # background_tasks.add_task(train_model, 'configs/object_detection.ini')
    return {
        'message': 'Model training has started',
    }


@router.post('/validate/')
async def validate(img: UploadFile = File(...)):
    from auto_train.segmentation.infer import infer
    image = Image.open(BytesIO(img.file.read()))
    logger.info(img.filename)
    img_path = f'test_images/{img.filename}'
    image.save(img_path)
    return infer(img_path)
