import typer

app = typer.Typer()

ALLOWED_TASKS = [
    'classification',
    'object_detection',
    'segmentation'
]

@app.command()
def train(task=None, config=None):
    assert task in ALLOWED_TASKS, f'Task should be one of {ALLOWED_TASKS}'
    print()
    if task == 'classification':
        import classification
        classification.train_model(config)

if __name__ == '__main__':
    app()
