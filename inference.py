 # to install the yolact package, run:
# cd yolact_pkg
# python -m pip install -e .
from yolact_pkg.data.config import Config
from yolact_pkg.data.config import resnet101_rgbd_backbone
from yolact_pkg.yolact import Yolact
from yolact_pkg.train import train
from yolact_pkg.eval import annotate_img
from torchinfo import summary


import cv2
import torch
import numpy as np

MEANS = (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453)
STD = (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024)

if __name__ == '__main__':
    
    print("dir()", dir())
    
    
    dataset = Config({
        'name': 'Base Dataset',

        # On Linux ubuntu
        # # Training images and annotations
        # 'train_images': '/media/hdd7/sblaue/coco_gaps_devices_RGBD/train/',
        # 'train_info':   '/media/hdd7/sblaue/coco_gaps_devices_RGBD/train/_train.json',

        # # # Validation images and annotations.
        # 'valid_images': '/media/hdd7/sblaue/coco_gaps_devices_RGBD/val/',
        # 'valid_info':   '/media/hdd7/sblaue/coco_gaps_devices_RGBD/val/_val.json',

        ## Loacl on Mac
        #Training images and annotations
        'train_images': '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/Data/temp_coco/train/',
        'train_info':   '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/Data/temp_coco/train/_train.json',

        # # Validation images and annotations.
        'valid_images': '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/Data/temp_coco/val/',
        'valid_info':   '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/Data/temp_coco/val/_val.json',

        # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
        'has_gt': True,

        # A list of names for each of you classes.
        'class_names': ('gap'),

        # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
        # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
        # If not specified, this just assumes category ids start at 1 and increase sequentially.
        'label_map': None,

    })
    
    config_override = {
        'name': 'yolact_base',

        # Dataset stuff
        'dataset': dataset,
        'num_classes': len(dataset.class_names) + 1,

        # Image Size
        'max_size': 512,
        
        # On Linux
        # 'save_path': '/media/hdd7/sblaue/weights/',

        # On Mac
        'save_path': '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/yolact/weights/',
        
        # we can override args used in eval.py:        
        'score_threshold': 0.1,
        'top_k': 10,

        'load_strict': False,

        #RGBD
        'rgbd':True,

        'MEANS': (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453),
        'STD': (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024),

        'augment_photometric_distort': False,
        'backbone': resnet101_rgbd_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug

        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]]}),

    }
    
    # we can override training args here:
    training_args_override = {
        "batch_size": 8,
        "save_interval": -1, # -1 for saving only at end of the epoch
        "cuda": False,
        "mps": False
        # "resume": 
    }
    
    yolact = Yolact(config_override)
    
    #summary(yolact,input_size=(8, 4, 512, 512) )
    
    # Make sure the current PyTorch binary was built with MPS enabled
    print(torch.backends.mps.is_built())
    # And that the current hardware and MacOS version are sufficient to
    # be able to use MPS
    print(torch.backends.mps.is_available())
 
 ###########################################
 #  Inference on Mac                     #
###########################################

    override_eval_config = Config({
        'cuda' : False,
        'top_k': 10,
        'score_threshold': 0.1,
        'display_masks': True,
        'display_fps' : False,
        'display_text': False,
        'display_bboxes': False,
        'display_scores': True,
        'save_path': '/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/yolact/weights/',
        'MEANS': (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453),
        'STD': (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024),
        
    })
    
    yolact.eval()

    # On Mac 
    weights = 'training_2022-06-06-RGBD_final/yolact_base_118_1309.pth'
    yolact.load_weights(override_eval_config.save_path + weights)


    devices_train = ['hca1','hca2','hca5','hca6','hca8','hca11','hca12'] 

    devices_val = ['hca0', 'hca3', 'hca7']

    devices_test = ['hca4', 'hca9']

    import glob
    PATH = "/Users/simonblaue/ownCloud/Bachelorarbeit/gap-detection/Data/temp_coco/train/"
    all_device_img_list = list(glob.iglob(PATH + "**/PNGImages/**.png", recursive=True ))

    import random
    for i in range(10):
        img_path = random.choice(all_device_img_list)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,:3]
        img_old = img
        frame, classes, scores, boxes, masks = yolact.infer(img_path)

        header = img_path.replace(PATH, '').replace('.png','').replace('PNGImages/','')

        annotated_img = annotate_img(frame, classes, scores, boxes, masks, override_args=override_eval_config)

        img = np.hstack((annotated_img,img))
        cv2.namedWindow(header, cv2.WINDOW_NORMAL)
        cv2.imshow(header,img)
        #cv2.imwrite("/Users/simonblaue/ownCloud/Bachelorarbeit/Figures/yolact/RGBD/unknown devices/" +device+".RGB"+str(number)+".png", img )

        cv2.waitKey(0)
        cv2.destroyAllWindows()