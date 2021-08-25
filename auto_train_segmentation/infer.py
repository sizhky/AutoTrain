from torch_snippets import *
from icevision.all import *
from auto_train_object_detection.custom_functions import *
from torch_snippets.registry import registry, Config, AttrDict

config = Config().from_disk('config.ini')
config = AttrDict(registry.resolve(config))

image_size = config.architecture.size
class_map = ClassMap(config.project.classes)

assert config.architecture.model_type.count('.', 1), "Architecture should look like <base>.<model>"
extra_args = config.architecture.extra_args
a, b = config.architecture.model_type.split('.')
model_type = getattr(getattr(models, a), b)
backbone = getattr(model_type.backbones, config.architecture.backbone)(config.architecture.pretrained)
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(class_map), **extra_args)

from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
yolo_path = config.training.scheme.output_path
load_torch_model_weights_to(model, yolo_path, device='cpu')


from collections import namedtuple
infer_tfms = config.testing.preprocess
print(infer_tfms)
Pred = namedtuple('Pred', ['bbs','labels'])

image_extns = ['jpg','jpeg','png']
def predict_on_folder_of_images(folder):
    fpaths = []
    for extn in image_extns:
        fpaths += P(folder).Glob(f'*.{extn}')
    imgs = [read(f, 1) for f in fpaths][:2]
    logger.info(f'Found {len(imgs)} images')
    infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=class_map)

    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model=model, infer_dl=infer_dl, keep_images=True)

    bboxes = [pred.pred.detection.bboxes for pred in preds]
    bboxes = [[(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) for bbox in bboxlist] for bboxlist in bboxes]
    bboxes = [[BB(int(x), int(y), int(X), int(Y)) for (x,y,X,Y) in bboxlist] for bboxlist in bboxes]
    shapes = [im.shape for im in imgs]
    pads = [((max(W, H) - min(W, H)) // 2) for H,W,_ in shapes]
    ws = [sh[1] for sh in shapes]
    bboxes = [[bb.remap((384,384), (ws[ix], ws[ix])) for bb in bblist] for ix, bblist in enumerate(bboxes)]
    bboxes = [[BB(x,y-pads[ix],X,Y-pads[ix]) for (x,y,X,Y) in bbs] for ix,bbs in enumerate(bboxes)]

    labels = [pred.pred.detection.labels for pred in preds]
    preds = lzip(bboxes, labels)
    preds = [Pred(*pred) for pred in preds]
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