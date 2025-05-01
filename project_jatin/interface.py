# interface.py

# model class
from model import EmotionCNN as TheModel

# training function
from train import train_emotion_model as the_trainer

# inference function
from predict import predict_emotion as the_predictor

# dataset class & loader
from dataset import FER2013Dataset as TheDataset
from dataset import fer_dataloader as the_dataloader

# hyperparameters
from config import batchsize as the_batch_size
from config import epochs as total_epochs
