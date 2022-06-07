from trainer import Trainer

if __name__ == "__main__":
    img_dir = 'male_female_face_images/'
    trainer = Trainer(img_dir, load=True)
    trainer.test()