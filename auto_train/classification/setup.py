import os
from torch_snippets.paths import writelines
from torch_snippets.registry import *
import typer

app = typer.Typer()

bash_file = '''
git clone https://github.com/rwightman/pytorch-image-models timm
cd timm
git checkout 54e90e8
git reset --hard
cd ..
export BASE_DATA_DIR=BASE_DATA_DIRECTORY
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=True

trap cleanup INT
cleanup() {
    echo "Killing label studio"
    kill "$label_studio_pid"
    echo "Killing label studio ml"
    kill "$label_studio_ml_pid"
    python auto_train_classification/config_to_timm.py
}

mkdir -p MODEL_DIRECTORY
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $PYTHONPATH
label-studio &
label_studio_pid=$!
echo "Label Studio lauched @ $label_studio_pid"
label-studio-ml init APP_NAME --script SCRIPT_PATH
label-studio-ml start ./APP_NAME
label_studio_ml_pid=$!
echo "Label Studio ML lauched @ $label_studio_ml_pid"

wait
'''

@app.command()
def main(config_path=os.environ['CONFIG']):
    config = Config().from_disk(config_path)
    config = AttrDict(config)
    file = (bash_file
        .replace('BASE_DATA_DIRECTORY', config.label_studio.base_data_dir)
        .replace('APP_NAME', config.label_studio_ml.app_name)
        .replace('SCRIPT_PATH', config.label_studio_ml.script_path)
        .replace('MODEL_DIRECTORY', config.project.model_directory)
    )
    file = [l for l in file.splitlines() if l.strip()!='']
    writelines(file, 'auto_train_classification/setup.sh')
    import rich
    from rich.panel import Panel
    info = '''Setup Done!
    Run
    $ [green]chmod +x auto_train_classification/setup.sh[/green]
    $ [green]./auto_train_classification/setup.sh[/green]
    in termial to start label studio labelling
    '''
    info = '\n'.join([l.strip() for l in info.splitlines()])
    rich.print(Panel.fit(info, border_style="red"))
    

if __name__ == '__main__':
    app()
