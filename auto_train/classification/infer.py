from torch_snippets import sys, P
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose, show, pd

from auto_train.classification.model import ClassificationModel

task = ClassificationModel(train_id='1')
learn, config, model = task.learn, task.config, task.model

weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path)

# def infer(folder='/home/yyr/data/imagenette2-160/val/n03394916/'):
def infer(folder='/home/yyr/code/AutoTrain/test_images'):
    fpaths = []
    image_extns = ['png','jpg','jpeg']
    image_extns = image_extns + [i.upper() for i in image_extns]
    for extn in image_extns:
        fpaths += P(folder).Glob(f'*.{extn}')
    fpaths = choose(fpaths, 2)
    output = []
    for f in fpaths:
        pred, _, cnf = learn.predict(f)
        conf = max(cnf)
        # show(read(f, 1), title=f'{pred} @ {conf*100:.1f}%')
        output.append((str(f.resolve()), pred, f'{conf*100:.2f}%'))
    output = pd.DataFrame(output, columns=['file', 'prediction', 'confidence'])
    return output.to_dict(orient='records')

