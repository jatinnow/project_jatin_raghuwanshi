# config.py

# Training hyperparameters
batchsize     = 64
epochs        = 20
learning_rate = 0.001

# Image input size
resize_x       = 48   # width
resize_y       = 48   # height
input_channels = 1    # 1 for grayscale, 3 for RGB

# Emotion labels (useful for indexing or decoding)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotion_labels)
