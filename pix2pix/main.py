from trainer import Trainer
trainer = Trainer(load=True)
val_dir = 'images'
trainer.eval(val_dir)