from torch_snippets import makedir, parent, logger, plt

def find_best_learning_rate(task):
    learn = task.learn
    config = task.config
    # with learn.no_bar():
    if 1:
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
