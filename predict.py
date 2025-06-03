import numpy as np
import tensorflow as tf
from PIL import Image

# Modeli sadece 1 defa yükle
interpreter = tf.lite.Interpreter(model_path="models/classifier_model.tflite")
interpreter.allocate_tensors()

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def predict_image_class(image_path):
    input_data = preprocess_image(image_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    prediction_index = np.argmax(output_data)
    confidence = float(np.max(output_data))

    class_names = ['Negative', 'İskemik İnme', 'Kanamalı İnme']
    return class_names[prediction_index], confidence
