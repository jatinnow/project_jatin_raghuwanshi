# dataset.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

class FER2013Dataset:
    def __init__(self, root_dir, subset):
        # subset is 'train' or 'test'
        self.dir = os.path.join(root_dir, subset)
        self.datagen = ImageDataGenerator(rescale=1./255)

    def get_generator(self):
        return self.datagen.flow_from_directory(
            self.dir,
            target_size=(config.resize_x, config.resize_y),
            color_mode='grayscale' if config.input_channels==1 else 'rgb',
            class_mode='sparse',
            batch_size=config.batchsize,
            shuffle=(subset=='train')
        )

def fer_dataloader(root_dir, subset):
    ds = FER2013Dataset(root_dir, subset)
    return ds.get_generator()
