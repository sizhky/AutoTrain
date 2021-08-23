# Step by step instrcutions

### 0. Initial Setup

1. install requirements
```bash
$ pip install -r requirements.txt
```
2.  Modify settings in `config.ini` file as well as the `preprocessing/head` in `auto_train_classification/custom_functions.py`

### 1. Create setup for starting label-studio using config.ini
```bash
$ python auto_train_classification/setup.py
```

### 2. start labelling and online-training
```bash
$ ./setup.sh 
```
Once done download the labels file to local Ctrl+C to exit

### 3. below command will copy images into the right folder structure for training
```bash
$ python auto_train_classification/setup_images.py config.ini ~/Downloads/project-label-studio-dump.csv
```

### 4. train in timm
```bash
$ python timm/train.py -c config.ini PROJECT/data/train
```

### 5. predict
```bash
$ python auto_train_classification/infer_classification.py path/to/folder/of/images
```
