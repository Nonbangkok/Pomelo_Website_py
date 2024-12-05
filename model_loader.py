# model_loader.py
from tensorflow.keras.models import load_model

def load_trained_model():
    model = load_model('model.h5', compile=False)
    print("Model loaded successfully.")
    return model

model = load_trained_model()