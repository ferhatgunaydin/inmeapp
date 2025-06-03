import cv2
import os
import uuid
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from predict import predict_image_class
from segment_predict import load_segment_model, predict_segmentation, create_overlay

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OVERLAY_FOLDER'] = 'static/overlays'
app.config['MASK_FOLDER'] = 'static/masks'
app.config['MODEL_PATH'] = 'models/segmenter_model.tflite'

segment_model = load_segment_model(app.config['MODEL_PATH'])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    overlay_path = None
    mask_path = None
    uploaded_path = None

    if request.method == 'POST':
        file = request.files['file']
        hasta_id = request.form['hasta_id']
        isim = request.form['isim']

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            uploaded_path = image_path

            prediction_label, confidence = predict_image_class(image_path)
            result = f"Tahmin: {prediction_label}, Güven: {confidence:.2%}"

            overlay_path = None
            mask_path = None

            if prediction_label in ['İskemik İnme', 'Kanamalı İnme']:
                red_mask, binary_mask = predict_segmentation(image_path, segment_model)
                mask_filename = f"mask_{uuid.uuid4().hex}.png"
                overlay_filename = f"overlay_{uuid.uuid4().hex}.png"
                mask_path = os.path.join(app.config['MASK_FOLDER'], mask_filename)
                overlay_path = os.path.join(app.config['OVERLAY_FOLDER'], overlay_filename)
                cv2.imwrite(mask_path, red_mask)
                create_overlay(image_path, binary_mask, overlay_path)

            with open('predictions.csv', 'a', encoding='utf-8') as f:
                f.write(f"{hasta_id},{isim},{prediction_label},{confidence:.2%},{mask_path},{overlay_path}\n")

    return render_template('index.html', result=result, uploaded=uploaded_path, mask=mask_path, overlay=overlay_path)

@app.route('/gecmis')
def gecmis():
    df = pd.read_csv('predictions.csv', names=["hasta_id", "isim", "tahmin", "güven", "mask_path", "overlay_path"])
    hasta_id = request.args.get("hasta_id")
    sonuc = request.args.get("tahmin")

    if hasta_id:
        df = df[df["hasta_id"].astype(str).str.contains(hasta_id)]
    if sonuc:
        df = df[df["tahmin"].str.contains(sonuc)]

    return render_template("gecmis.html", records=df.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)
