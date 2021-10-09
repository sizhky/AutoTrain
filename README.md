# AutoTrain

Have you ever wondered if there is a framework that uses SOTA techniques to train Deep Learning models that needs (almost) no coding?  
⚡️ You are in the right place! 

**Train your deep learning models with nothing more than a config file (and a supporting python file if needed).**

That's right, with almost no code you can train state of the art models for 
* Image Classification,
* Object Detection, and
* Segmentation

All you have to do is create a copy of one of the `ini` files found in the `configs` folder and change dataset details and the hyperparameters as per your choice. 

Training on a config file is just one line of code

```bash
$ python main.py --task=[classification/object_detection/segmentation] --config=[configs/*.ini]
```

At the end of training you will have a weights file generated in the corresponding projects folder that can be used for inference.

For testing purposes, there is a FastAPI server that can serve your models as APIs (currently serving one image per request). Just go to the corresponding task's `infer.py` file and change the config to point to your own config file

## Features
* Every parameter and hyper-parameter is visible in the config file
  * maintains experiment transparency
  * improves comprehension of the experiment
* Auto download dataset from a URL if data is not present
* Context aware config means you can reuse variables within `ini` file for easier maintenance
* All the files (data, models) will be generated in locations of your choice
* Registering functions as strings will let you replace standard functions with your own custom functions in the pipeline
* Train and Test using a single call in terminal
* Expose an endpoint just by using the config file

## Blog Posts
* [Introduction](https://sizhky.github.io/posts/2021/10/auto-train.html)
* [Ingredients](https://sizhky.github.io/posts/2021/10/auto-train-config.html)
* [Recipes](https://sizhky.github.io/posts/2021/10/auto-train-boiler-plate.html)
* [Use Cases](https://sizhky.github.io/posts/2021/10/auto-train-use-cases.html)

### Credits

#### Classification
Uses [fastai](https://github.com/fastai/fastai) and [timm](https://github.com/rwightman/pytorch-image-models) libraries to expose one line functions that can create architectures by using a single string

#### Object Detection and Segmentation
Uses [fastai](https://github.com/fastai/fastai) and [icevision](https://github.com/airctic/icevision) libraries to expose similar functionality

#### Other awesome libraries used
* [FastAPI](https://github.com/tiangolo/FastAPI)  
* [Typer](https://github.com/tiangolo/Typer)  
* [torch-snippets](https://github.com/sizhky/torch_snippets)  
