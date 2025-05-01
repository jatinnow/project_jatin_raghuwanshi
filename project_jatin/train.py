# train.py

import os
from model import EmotionCNN
from dataset import fer_dataloader
import config
import tensorflow as tf

def train_emotion_model():
    # prepare data
    train_gen = fer_dataloader('data', 'train')
    val_gen   = fer_dataloader('data', 'test')

    # instantiate & compile
    model = EmotionCNN()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # callbacks: save final weights to checkpoints/final_weights.pth
    # (we’ll save in TensorFlow native format but rename .pth)
    ckpt_path = os.path.join('checkpoints', 'final_weights.keras')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, save_best_only=True, monitor='val_accuracy'
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    # train
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.epochs,
        callbacks=[ckpt, es]
    )

    # finally, save (this will overwrite with the best)
    model.save(ckpt_path)
