 # to install the yolact package, run:
# cd yolact_pkg
# python -m pip install -e .
from yolact_pkg.data.config import Config
from yolact_pkg.yolact import Yolact
from yolact_pkg.eval import annotate_img
from settings import *

import cv2
import numpy as np
import glob
import random

if __name__ == '__main__':
    
    # config_override.update(override_eval_config.)
    # print(config_override)
    # init net with config from settings.py
    yolact = Yolact(config_override)
    
 ###########################################
 #  Inference on Mac                     #
###########################################

    # Setting net in eval mode (train=False)
    yolact.eval()

    # Reloade pretrained model
    # model_weights = '/Users/simonblaue/Desktop/YOLACT_RGBD/models/RGBD/yolact_base_118_1309.pth' #RGBD
    model_weights = '/Users/simonblaue/Desktop/YOLACT_RGBD/models/RGB/yolact_base_113_1254.pth' #RGB
    yolact.load_weights(model_weights)

    # If you want to get images from a camera stream make sure they are 4 channels with Depth renormalized to 1..255 (no zeros)
    train_images = list(glob.iglob(dataset.train_images + '**/PNGImages/**.png', recursive=True))
    val_images = list(glob.iglob(dataset.valid_images + '**/PNGImages/**.png', recursive=True))

    #image_name = '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/val/PNGImages/hca0.RGBD3.png' # Set to none if you want to see some random results
    #image_name = None

    image_name = 'save_all' # uncomment to save inference for all train and val images
    save_inference_vals = '/Users/simonblaue/ownCloud/Bachelorarbeit/Figures/yolact/RGBD/val_inference/'
    save_inference_train = '/Users/simonblaue/ownCloud/Bachelorarbeit/Figures/yolact/RGBD/train_inference/'

    print(f"Found {len(train_images)} train images and {len(val_images)}.")
    if image_name == None:
        for i in range(5):
            img_path = random.choice(val_images)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,:]
            
            frame, classes, scores, boxes, masks = yolact.infer(img_path)
            annotated_img = annotate_img(frame, classes, scores, boxes, masks, override_args=override_eval_config)

            img = np.hstack((annotated_img,img[:,:,:3]))
            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.imshow('test',img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif image_name == 'save_all':
        for img_path in train_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,:]
            
            frame, classes, scores, boxes, masks = yolact.infer(img_path)
            annotated_img = annotate_img(frame, classes, scores, boxes, masks, override_args=override_eval_config)

            img = np.hstack((annotated_img,img[:,:,:3]))
            name = img_path.replace(dataset.train_images,'').replace('PNGImages/','')
            filepath = save_inference_train + name
            cv2.imwrite(filepath, img)
        for img_path in val_images:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,:]
            
            frame, classes, scores, boxes, masks = yolact.infer(img_path)
            annotated_img = annotate_img(frame, classes, scores, boxes, masks, override_args=override_eval_config)

            img = np.hstack((annotated_img,img[:,:,:3]))
            name = img_path.replace(dataset.valid_images,'').replace('PNGImages/','')
            filepath = save_inference_vals + name
            cv2.imwrite(filepath, img)
        print(f"Wrote {len(train_images)+ len(val_images)} files.")

    else:
        img = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)[:,:,:]
            
        frame, classes, scores, boxes, masks = yolact.infer(image_name)
        annotated_img = annotate_img(frame, classes, scores, boxes, masks, override_args=override_eval_config)

        img = np.hstack((annotated_img,img[:,:,:3]))
        cv2.namedWindow('Window name', cv2.WINDOW_NORMAL)
        cv2.imshow('Window name',img)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()