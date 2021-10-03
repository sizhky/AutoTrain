from icevision.all import Dataset
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose

from auto_train.segmentation.model import SegmentationModel

task = SegmentationModel(config='configs/segmentation.ini', inference_only=True)
config, model = task.config, task.model
model_type, parser = task.model_type, task.parser

weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path)

def infer(fpath):
    fpaths = [fpath]
    imgs = [read(f, 1) for f in fpaths]
    logger.info(f'Found {len(imgs)} images')

    infer_tfms = config.testing.preprocess

    infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=parser.class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
    # show_preds(preds=preds)
    return preds
