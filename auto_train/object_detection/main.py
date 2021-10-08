from torch_snippets import rand, PIL, np, Image, show, read
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from io import BytesIO
import os
from loguru import logger

router = APIRouter()

@router.post('/train/')
async def train(background_tasks: BackgroundTasks):
    from auto_train.object_detection.train import train_model
    # background_tasks.add_task(train_model, 'configs/object_detection.ini')
    return {
        'message': 'Model training has started',
    }


@router.post('/validate/')
async def validate(img: UploadFile = File(...)):
    from auto_train.object_detection.infer import infer
    image = Image.open(BytesIO(img.file.read()))
    logger.info(img.filename)
    img_path = f'test_images/{img.filename}'
    image.save(img_path)
    preds = infer(img_path)
    show(
        read(img_path, 1),
        bbs=preds['bbs'],
        texts=preds['labels'],
        save_path='test_images/object_detection.png'
        )

    return FileResponse('test_images/object_detection.png')
