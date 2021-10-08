from torch_snippets import sys, P
from torch_snippets import load_torch_model_weights_to, logger, read, P, choose, show, pd

import auto_train.classification.custom_functions
from auto_train.classification.model import ClassificationModel

task = ClassificationModel(config='configs/classification_rooftops.ini')
learn, config, model = task.learn, task.config, task.model
weights_path = config.training.scheme.output_path
load_torch_model_weights_to(model, weights_path)

def infer(img_path):
    logger.info(f'received {img_path} for classification')
    pred, _, cnf = learn.predict(img_path)
    prediction = {
        'prediction': pred,
        'confidence': f'{max(cnf)*100:.2f}%'
    }
    if 'postprocess' in dir(config.testing):
        return config.testing.postprocess(prediction)
    else:
        return prediction
