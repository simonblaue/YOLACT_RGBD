# YOLACT_RGBD

This is an extension from YOLACT https://github.com/dbolya/yolact.git to include depth data in the model.

## Get started
First download this repo, then install the yolact package from this in your pytorch with cuda environment. To set up such an envirement install the dependencies with the requirements file and an according cuda version.

To install the yolact package, run:
   cd yolact_pkg
   python -m pip install -e .



## Models and Datasets
If you want to retrain the RGBD model you need the pretrained backbone weights from resnet1010 found <a href='https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view'>here </a>. This file has to be put in ./weights in this repo.

For datasets or pre trained models send me a message.

## How to use

### Options to change in settings.py

In the dataset Config options change:
   - train_images path
   - train_info path
   - valid_images path
   - valid_info path

In config_override change:
   -  save_path to the weights folder in this repo
   -  backbone, MEANS, STD and augment_photometric_distort if you wan to train in RGB


## Labelme for dataset creation
 
The dataset was labeld with labelme in RGB. 

Then the depth data was appended as a fourth channel to the images. 

Finaly use the labelme export function to export annotaions in COCO file format using the command ...

