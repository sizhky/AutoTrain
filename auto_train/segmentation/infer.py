from icevision.all import Dataset
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose, resize, np, PIL, show

from auto_train.segmentation.model import SegmentationModel

task = SegmentationModel(config='configs/segmentation.ini', inference_only=True)
config, model = task.config, task.model
model_type, parser = task.model_type, task.parser

weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path)

def infer(fpath):
    img = read(fpath, 1)

    infer_tfms = config.testing.preprocess

    infer_ds = Dataset.from_images([img], infer_tfms, class_map=parser.class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    pred = model_type.predict_from_dl(model, infer_dl, keep_images=True)[0]

    h, w = read(fpath).shape
    mask = pred.pred.detection.mask_array.to_tensor()
    mask = mask.sum(0) / len(mask)
    mask = mask[...,None].repeat(1,1,3)
    mask = resize((mask > 0.2).cpu().detach().numpy().astype(np.uint8)*255, (h,w))
    nz_pxls = np.where(mask > 0)
    _img = img.mean(-1)[...,None].repeat(3, 2)
    _img[nz_pxls] = img[nz_pxls]
    return _img.astype(np.uint8), mask
