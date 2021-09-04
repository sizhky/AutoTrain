from torch_snippets import (
    sys, choose, resize,
    load_torch_model_weights_to,
    readPIL, logger, BB,
    lzip, show, P, read, PIL, np
)
sys.path.append(str(P().resolve()))
from auto_train_object_detection.model import (
    config, Dataset, model_type, 
    model, class_map, show_preds, parser,
    convert_raw_predictions
)

yolo_path = config.training.scheme.output_path
load_torch_model_weights_to(model, yolo_path, device='cpu')

from collections import namedtuple
infer_tfms = config.testing.preprocess

Pred = namedtuple('Pred', ['bbs','labels'])

def preds2bboxes(preds):
    bboxes = [pred.pred.detection.bboxes for pred in preds]
    bboxes = [[(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) for bbox in bboxlist] for bboxlist in bboxes]
    bboxes = [[BB(int(x), int(y), int(X), int(Y)) for (x,y,X,Y) in bboxlist] for bboxlist in bboxes]
    return bboxes

def infer(fpaths=None, folder=None):
    if not fpaths:
        fpaths = []
        image_extns = ['png','jpg','jpeg']
        for extn in image_extns:
            fpaths += P(folder).Glob(f'*.{extn}')
        fpaths = choose(fpaths, 4)
    imgs = [read(f, 1) for f in fpaths]
    logger.info(f'Found {len(imgs)} images')

    infer_ds = Dataset.from_images(imgs, config.testing.preprocess, class_map=parser.class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
    show_preds(preds=preds, ncols=3)
    return preds

def post_process(preds, images):
    size = config.architecture.size
    bboxes = preds2bboxes(preds)
    shapes = [im.shape for im in images]
    pads = [((max(W, H) - min(W, H)) // 2) for H,W,_ in shapes]
    ws = [max(sh) for sh in shapes]
    bboxes = [[bb.remap((size,size), (ws[ix], ws[ix])) for bb in bblist] for ix, bblist in enumerate(bboxes)]
    # bboxes = [[BB(x,y-pads[ix],X,Y-pads[ix]) for (x,y,X,Y) in bbs] for ix,(sh, bbs) in enumerate(zip(shapes, bboxes))]
    _bboxes = []
    for ix, (sh, bbs) in enumerate(zip(shapes, bboxes)):
        w, h, _ = sh
        if w < h:
            _bboxes.append([BB(x,y-pads[ix],X,Y-pads[ix]) for (x,y,X,Y) in bbs])
        else:
            _bboxes.append([BB(x-pads[ix],y,X-pads[ix],Y) for (x,y,X,Y) in bbs])
    bboxes = _bboxes

    labels = [pred.pred.detection.labels for pred in preds]
    preds = lzip(bboxes, labels)
    preds = [Pred(*pred) for pred in preds]
    return preds

image_extns = ['jpg','jpeg','png']
def predict_on_folder_of_images(fpaths, 
    batch_size=config.testing.batch_size,
    detection_threshold=config.testing.detection_threshold,
    nms_iou_threshold=config.testing.nms_iou_threshold
    ):
    images = [read(image_path, 1) for image_path in fpaths]
    infer_ds = Dataset.from_images(
        images, config.testing.preprocess, 
        class_map=class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=batch_size, drop_last=False)
    preds = []
    for tensor, records in iter(infer_dl):
        x = tensor[0].cuda()
        output = model.eval()(x)
        pred = convert_raw_predictions(
            x, output[0], records,
            detection_threshold=detection_threshold,
            nms_iou_threshold=nms_iou_threshold
        )
        preds.extend(pred)

    preds = post_process(preds, images)
    output = lzip(fpaths, preds)
    return output

import typer
app = typer.Typer()

@app.command()
def show_predictions_on_folder_of_images(folder):
    for fpath, pred in predict_on_folder_of_images(folder):
        try:
            show(read(fpath, 1), bbs=pred.bbs, texts=pred.labels, fpath=f'/content/outputs/{fpath}')
        except Exception as e:
            logger.warning(f'Failed to show prediction on {fpath}\nError {e}')

if __name__ == '__main__':
    app()