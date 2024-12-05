from tensorflow.keras.models import load_model

def load_trained_model():
    model = load_model('model.h5')
    print("Model loaded successfully.")
    return model

model = load_trained_model()