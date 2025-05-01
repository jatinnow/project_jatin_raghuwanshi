# predict.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from model import EmotionCNN
import config

def predict_emotion(list_of_img_paths, weights_path='checkpoints/final_weights.keras'):
    # load model & weights
    model = EmotionCNN()
    model.load_weights(weights_path)

    # preprocess all images
    batch = []
    for p in list_of_img_paths:
        img = image.load_img(p,
            color_mode='grayscale' if config.input_channels==1 else 'rgb',
            target_size=(config.resize_x, config.resize_y))
        arr = image.img_to_array(img)/255.0
        batch.append(arr)
    batch = np.stack(batch)

    # predict
    probs = model.predict(batch)
    classes = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1) * 100
    return list(classes), list(confidences)
