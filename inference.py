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
    
    # init net with config from settings.py
    yolact = Yolact(config_override)
    
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

    # Reloade pretrained model
    weights = 'training_2022-06-06-RGBD_final/yolact_base_118_1309.pth'
    yolact.load_weights(override_eval_config.save_path + weights)

    train_images = list(glob.iglob(dataset.train_images + '**/PNGImages/**.png', recursive=True))
    val_images = list(glob.iglob(dataset.val_images + '**/PNGImages/**.png', recursive=True))

   
    for i in range(10):
        img_path = random.choice(train_images)
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