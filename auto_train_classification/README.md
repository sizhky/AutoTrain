# Step by step instrcutions

### 0. Initial Setup

1. install requirements
```bash
$ pip install -r requirements/classification.txt
```
2.  Modify settings in `config_classification.ini` file as well as the `preprocessing/head` in `auto_train_classification/custom_functions.py`
3. Set the environment variable `CONFIG` to point to the path `config_classification.ini` in shell

### 1. Create setup for starting label-studio using config.ini
```bash
$ python auto_train_classification/setup.py
```

### 2. Start labelling and online-training
```bash
$ chmod +x auto_train_classification/setup.sh
$ ./auto_train_classification/setup.sh 
```
1. Add a new label-studio project, upload your files and set the task as classification
2. Connect the ML model (the URL looks something that ends with 9090 port in the console STDOUT (http://192.168.0.125:9090/))
3. Deselect all the three options in ML setup
3. Manually label a few tasks, go back to Machine Learning setup and select the options
> Retrieve predictions when loading a task automatically
> Show predictions to annotators in the Label Stream and Quick View
4. Select start training. This will train a new ML model on the lables you have given with architecture based on your config file.
5. Once labelling is done download the labels file in **csv** format to appropriate folder and `Ctrl+C` to exit

### 3. Below command will copy images into the right folder structure for training
```bash
$ python auto_train_classification/setup_images.py ~/Downloads/project-label-studio-dump.csv
```

### 4. Train in timm
```bash
$ python timm/train.py -c config.ini PROJECT/data/train
```

### 5. Predict
```bash
$ python auto_train_classification/infer_classification.py path/to/folder/of/images
```
