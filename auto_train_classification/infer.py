from torch_snippets import sys, P
sys.path.append(str(P().resolve()))
from auto_train_classification.model import config, learn, model
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose
weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path)

def infer(folder):
    fpaths = []
    image_extns = ['png','jpg','jpeg']
    image_extns = image_extns + [i.upper() for i in image_extns]
    for extn in image_extns:
        fpaths += P(folder).Glob(f'*.{extn}')
    fpaths = choose(fpaths, 4)
    output = []
    for f in fpaths:
        pred, _, cnf = learn.predict(f)
        conf = max(cnf)
        output.append((f, pred, conf))
    return output
