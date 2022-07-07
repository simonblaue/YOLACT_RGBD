from yolact_pkg.yolact import Yolact
from yolact_pkg.train import train
from settings import *

## IMPORTATNT !! ##

# 
#   To train yolact you need cuda and pytorch installed. See requirements.txt!
# 

if __name__ == '__main__':
    
    print("dir()", dir())
    
    # Init net with config from settings.py
    yolact = Yolact(config_override)
    
    ###########################################
    # Training                                #
    ###########################################
    
    print("run training...")
    train(yolact, training_args_override)
   