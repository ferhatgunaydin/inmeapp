import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def load_segment_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_segmentation_input(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict_segmentation(image_path, interpreter):
    img_array = preprocess_segmentation_input(image_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # 1. Kanalı al
    mask = output_data[:, :, 0]

    # 2. Normalize et: 0-255 arası
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-8)
    mask = (mask * 255).astype(np.uint8)

    # 3. Eğer çok koyu ise eşik uygula
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    return binary_mask, mask  # ilki eşikli, ikincisi orijinal normalize maske

def create_overlay(image_path, mask, output_path):
    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (128, 128))

    # Maske eşikli, tek kanallı → renklendir (kırmızı)
    color_mask = np.zeros_like(orig)
    color_mask[:, :, 2] = mask  # kırmızı kanal

    # Daha belirgin olması için alpha oranını artır
    overlay = cv2.addWeighted(orig, 0.5, color_mask, 0.5, 0)

    cv2.imwrite(output_path, overlay)
