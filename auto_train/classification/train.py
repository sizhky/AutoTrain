from torch_snippets import sys, P
from auto_train.classification.model import learn, config, model
from torch_snippets import makedir, parent, logger, plt
import typer

app = typer.Typer()

def find_best_learning_rate():
    with learn.no_bar():
        suggested_lrs = learn.lr_find(show_plot=False)
    recorder = learn.recorder
    skip_end = 5
    lrs    = recorder.lrs    if skip_end==0 else recorder.lrs   [:-skip_end]
    losses = recorder.losses if skip_end==0 else recorder.losses[:-skip_end]
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    makedir(config.project.location)
    fig.savefig(f'{config.project.location}/find_lr_plot.png')
    logger.info(f'LR Plot is saved at {config.project.location}/find_lr_plot.png')
    try:
        logger.info(f'Suggested LRs: {suggested_lrs.lr_min} and {suggested_lrs.lr_steep}')
        return max(suggested_lrs.lr_min, suggested_lrs.lr_steep)
    except:
        return suggested_lrs.valley

@app.command()
def train_model(lr:float=None):
    from torch_snippets import load_torch_model_weights_to, save_torch_model_weights_from, makedir
    try:
        load_torch_model_weights_to(model, config.training.scheme.resume_training_from)
    except:
        logger.info('Training from scratch!')
        
    training_scheme = config.training.scheme

    lr = lr if lr is not None else find_best_learning_rate()
    logger.info(f"Using lr: {lr}")
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
