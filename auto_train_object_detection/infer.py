from torch_snippets import *
from icevision.all import *

image_size = 384
class_map = flatten([[f'{i}_ticked', f'{i}_not_ticked'] for i in range(1, 22)])
class_map = ClassMap(class_map)

extra_args = {}
model_type = models.ultralytics.yolov5
backbone = model_type.backbones.small
# The yolov5 model requires an img_size parameter  
# The efficientdet model requires an img_size parameter
extra_args['img_size'] = image_size
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(class_map), **extra_args) 


from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
yolo_path = P.home() / 'Downloads/YOLO.pth'
load_torch_model_weights_to(model, yolo_path)


from collections import namedtuple
infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=384), tfms.A.Normalize()])
Pred = namedtuple('Pred', ['bbs','labels'])

def predict_on_folder_of_images(folder):
    fpaths = P(folder).Glob('*.png')
    imgs = [read(f, 1) for f in fpaths]
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

folder = P.home()/'Desktop/dental-test'
for fpath, pred in predict_on_folder_of_images(folder):
    try:
        show(read(fpath, 1), bbs=pred.bbs, texts=pred.labels)
    except:
        ...