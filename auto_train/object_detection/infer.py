from collections import namedtuple
from torch_snippets import (
    sys,
    load_torch_model_weights_to,
    read, logger, BB,
    lzip, show, P, choose
)
from icevision.all import Dataset

from auto_train.object_detection.model import ObjectDetection

Pred = namedtuple('Pred', ['bbs','labels'])

task = ObjectDetection(config='configs/object_detection.ini', inference_only=True)
config, model = task.config, task.model
model_type, class_map = task.model_type, task.class_map

weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path, device='cpu')

infer_tfms = config.testing.preprocess

def preds2bboxes(preds):
    bboxes = [pred.pred.detection.bboxes for pred in preds]
    bboxes = [[(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) for bbox in bboxlist] for bboxlist in bboxes]
    bboxes = [[BB(int(x), int(y), int(X), int(Y)) for (x,y,X,Y) in bboxlist] for bboxlist in bboxes]
    return bboxes

def infer(fpath):
    logger.info(f'received {fpath} for object detection')
    imgs = [read(fpath, 1)]
    logger.info(f'Found {len(imgs)} images')
    infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=class_map)

    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model=model, infer_dl=infer_dl, keep_images=True)

    bboxes = preds2bboxes(preds)
    sz = config.architecture.size
    bboxes = [[BB(x/sz,y/sz,X/sz,Y/sz) for x,y,X,Y in bbs] for img, bbs in zip(imgs, bboxes)]

    labels = [pred.pred.detection.labels for pred in preds]
    preds = lzip(bboxes, labels)
    preds = [Pred(*pred) for pred in preds]
    return preds[0]._asdict()

