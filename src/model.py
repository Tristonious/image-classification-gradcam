import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image


def load_model():
    """Load VGG16 with pretrained ImageNet weights."""
    return VGG16(weights='imagenet')


def load_and_preprocess(img_path):
    """
    Open an image, resize to 224x224, and return:
      orig_img  — PIL Image at original size (RGB)
      arr       — preprocessed numpy array ready for VGG16 inference
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(
            f"Image not found: {img_path}. Place it in the repo root.")
    orig_img = Image.open(img_path).convert('RGB')
    img = orig_img.resize((224, 224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return orig_img, arr


def predict_top3(model, arr):
    """Run inference and return top-3 decoded predictions."""
    preds = model.predict(arr)
    return decode_predictions(preds, top=3)[0]
