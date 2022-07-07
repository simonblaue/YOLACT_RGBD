from yolact_pkg.data.config import Config
from yolact_pkg.data.config import resnet101_rgbd_backbone 
from yolact_pkg.data.config import resnet101_three_input_layers_backbone
from yolact_pkg.data.config import resnet101_rgbd_bakcbone_one_layer

    
dataset = Config({
    'name': 'Base Dataset',
    # Training folder 
    'train_images': '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/train/',
    # Train annotaions json file
    'train_info':   '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/train/annotations.json',
    # Validation folder
    'valid_images': '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/val/',
    # Validation annotaions json file
    'valid_info':   '/Users/simonblaue/Desktop/YOLACT_RGBD/datasets/heat allocators/val/annotations.json',
    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,
    # A list of names for each of you classes.
    'class_names': ('gap'),
    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
   })
    
config_override = {
    'name': 'yolact_base',
    # Dataset stuff
    'dataset': dataset,
    'num_classes': len(dataset.class_names) +1,
    # Image Size
    'max_size': 512,
    
    'save_path': '/Users/simonblaue/Desktop/YOLACT_RGBD/weights/resnet101_reducedfc.pth',
    
    # we can override args used in eval.py:        
    'score_threshold': 0.1,
    'top_k': 10,
    
    # Change these for different nets
    'MEANS': (116.24457136111748,119.55194544312776,117.05760736644808,196.36951043344453),
    'STD': (1.4380884974626822,1.8110670756137501,1.5662493838264602,1.8686978397590024),
    'augment_photometric_distort': False,
    'backbone': resnet101_rgbd_backbone.copy({
        'selected_layers': list(range(1, 4)),
        'use_pixel_scales': True,
        'preapply_sqrt': False,
        'use_square_anchors': True, # This is for backward compatability with a bug
        'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
        'pred_scales': [[24], [48], [96], [192], [384]],
    }),
}

# we can override training args here:
training_args_override = {
    "batch_size": 8,
    "save_interval": -1, # -1 for saving only at end of the epoch
}
