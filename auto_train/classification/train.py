import typer
from torch_snippets import (
    P, sys, logger, parent,
    load_torch_model_weights_to,
    save_torch_model_weights_from,
    makedir)

from auto_train.classification.model import ClassificationModel
from auto_train.common import find_best_learning_rate

app = typer.Typer()

@app.command()
def train_model(config: P):

    task = ClassificationModel(config)
    learn = task.learn
    model = task.model
    config = task.config
    training_scheme = config.training.scheme

    try:
        load_torch_model_weights_to(model, config.training.scheme.resume_training_from)
    except:
        logger.info('Training from scratch!')

    lr = find_best_learning_rate(task)
    logger.info(f"Using learning rate: {lr}")
    with learn.no_bar():
        print(["Epoch, Train Loss, Validation Loss, Validation Accuracy, Time"])
        learn.fine_tune(
            training_scheme.epochs, 
            lr, 
            freeze_epochs=training_scheme.freeze_epochs
        )

    makedir(parent(training_scheme.output_path))
    save_torch_model_weights_from(model, training_scheme.output_path)
    
if __name__ == '__main__':
    app()
