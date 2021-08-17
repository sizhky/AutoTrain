# Step by step instrcutions

```bash
# 0. install requirements
$ pip install -r requirements.txt

# 0.5 Modify settings in config.ini file 
# as well as the preprocessing/head in
# auto_train/custom_functions.py

# 1. Create setup for starting label-studio using config.ini
$ python auto_train/setup.py

# 2. start labelling and online-training
$ ./setup.sh 
# Once done download the labels file to local
# Ctrl+C to exit

# 3. below command will copy images into the right folder structure for training
$ python auto_train/setup_images.py config.ini ~/Downloads/project-label-studio-dump.csv

# 4. train in timm
$ python timm/train.py -c config.ini
```