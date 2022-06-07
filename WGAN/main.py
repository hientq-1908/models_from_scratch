from trainer import Trainer
import os
if __name__=="__main__":
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint') 

    trainer = Trainer()
    trainer.train()