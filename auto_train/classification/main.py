from torch_snippets import rand
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from io import BytesIO
from PIL import Image
import os
from loguru import logger
os.environ['CONFIG'] = 'configs/classification_imagenette.ini'

router = APIRouter()

path = os.path.normpath(__file__)
IMG_ROOT = os.sep.join(path.split(os.sep)[:-1])

logger.info(IMG_ROOT)


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
async def validate(img: UploadFile = File(...)):
    from auto_train.classification.infer import infer
    image = Image.open(BytesIO(img.file.read()))
    logger.info(img.filename)
    img_path = f'{IMG_ROOT}/images/{img.filename}'
    image.save(img_path)

    return infer(img_path=img_path)
