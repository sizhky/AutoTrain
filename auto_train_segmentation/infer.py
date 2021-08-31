from auto_train_segmentation.model import config, model, tfms, model_type, Dataset, show_preds, class_map
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose
yolo_path = config.training.scheme.output_path
load_torch_model_weights_to(model, yolo_path)

from typer import Typer

app = Typer()

@app.command()
def infer(folder):
    fpaths = []
    image_extns = ['png','jpg','jpeg']
    for extn in image_extns:
        fpaths += P(folder).Glob(f'*.{extn}')
    imgs = [read(f, 1) for f in choose(fpaths, 4)]
    logger.info(f'Found {len(imgs)} images')

    infer_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(size=512),
        tfms.A.Normalize()
    ])

    infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
    show_preds(preds=preds, ncols=3)
    return preds

if __name__ == '__main__':
    app()