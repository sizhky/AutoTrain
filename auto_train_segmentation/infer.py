from torch_snippets import (
    load_torch_model_weights_to,
    read, logger, BB,
    lzip, show, P, choose, sys
)

sys.path.append(str(P().resolve()))
from auto_train_object_detection.model import (
    model, config, 
    Dataset, model_type, parser,
    tfms, show_preds
)

yolo_path = config.training.scheme.output_path
load_torch_model_weights_to(model, yolo_path, device='cpu')


from collections import namedtuple
infer_tfms = config.testing.preprocess
Pred = namedtuple('Pred', ['bbs','labels'])

image_extns = ['jpg','jpeg','png']
def infer(folder):
    fpaths = []
    for extn in image_extns:
        fpaths += P(folder).Glob(f'*.{extn}')
    imgs = [read(f, 1) for f in choose(fpaths, 4)]
    logger.info(f'Found {len(imgs)} images')

    infer_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(size=512),
        tfms.A.Normalize()
    ])

    infer_ds = Dataset.from_images(imgs, infer_tfms, class_map=parser.class_map)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1)
    preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)
    show_preds(preds=preds, ncols=3)
    return preds

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