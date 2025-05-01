# model.py

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import config

class EmotionCNN(Sequential):
    def __init__(self):
        super().__init__([
            Input(shape=(config.resize_x, config.resize_y, config.input_channels)),
            Conv2D(32,3, activation='relu'), MaxPooling2D(), Dropout(0.25),
            Conv2D(64,3, activation='relu'), MaxPooling2D(), Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'), Dropout(0.5),
            Dense(7, activation='softmax')
        ])
