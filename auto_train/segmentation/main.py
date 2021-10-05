from torch_snippets import rand, PIL, np, Image
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from fastapi.responses import StreamingResponse
from io import BytesIO
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
    masked_image, raw_mask = infer(img_path)

    PIL.Image.fromarray(masked_image.astype(np.uint8)).save('test_images/segmentation.png')
    def iterfile():  
        with open('test_images/segmentation.png', mode="rb") as file_like:  
            yield from file_like
    return StreamingResponse(iterfile(), media_type='image/png')
