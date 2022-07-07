# YOLACT_RGBD

This is an extension from YOLACT https://github.com/dbolya/yolact.git to include depth data in the model.

## Get started
First download this repo, then install the yolact package from this in your pytorch with cuda environment. 

To install the yolact package, run:
   cd yolact_pkg
   python -m pip install -e .

## Models and Datasets
If you want to retrain the RGBD model you need the pretrained backbone weights from resnet1010 found <a href='https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view'>here </a>. The path to these has to be added in the train.py file under
For pretrained models or datasets send me a message.

## How to use

### Train
For training use the train.py file. 

There you have to change the path to your dataset images (train and validation). The images have to be png files with the depth added as the alpha channel.

The annotaions files have to be in the COCO falvoured style for datasets, best created with Labelme (see labelme section)

### Inference

## Options important for RGBD


# Datasets

## Labelme 