
import cv2
import os
import uuid
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from predict import predict_image_class
from segment_predict import load_segment_model, predict_segmentation, create_overlay
from fpdf import FPDF

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
    df = pd.read_csv('predictions.csv', names=["hasta_id", "isim", "tahmin", "güven", "mask_path", "overlay_path"], skiprows=1)
    hasta_id = request.args.get("hasta_id")
    sonuc = request.args.get("tahmin")

    if hasta_id:
        df = df[df["hasta_id"].astype(str).str.contains(hasta_id)]
    if sonuc:
        df = df[df["tahmin"].str.contains(sonuc)]

    return render_template("gecmis.html", records=df.to_dict(orient="records"))

@app.route('/indir_csv')
def indir_csv():
    return send_file('predictions.csv', as_attachment=True)

@app.route('/indir_pdf/<hasta_id>')
def indir_pdf(hasta_id):
    import unicodedata
    from fpdf import FPDF

    def remove_accents(text):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

    df = pd.read_csv(
        'predictions.csv',
        names=["hasta_id", "isim", "tahmin", "güven", "mask_path", "overlay_path"],
        skiprows=1
    )

    filtered = df[df['hasta_id'].astype(str) == str(hasta_id)]
    if filtered.empty:
        return f"'{hasta_id}' ID'sine sahip kayıt bulunamadı.", 404

    row = filtered.iloc[0]

    pdf = FPDF()
    pdf.add_page()

    # Font yolları
    font_path_regular = os.path.join(app.root_path, "static", "fonts", "DejaVuSans.ttf")
    font_path_bold = os.path.join(app.root_path, "static", "fonts", "DejaVuSans-Bold.ttf")

    pdf.add_font('DejaVu', '', font_path_regular, uni=True)
    pdf.add_font('DejaVu', 'B', font_path_bold, uni=True)

    # Logo ve başlık
    logo_path = os.path.join(app.root_path, "static", "logo.png")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=85, y=10, w=40)
    pdf.set_y(55)

    pdf.set_font("DejaVu", 'B', 14)
    pdf.set_fill_color(52, 152, 219)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 10, "Hasta Tahmin Raporu", ln=True, align='C', fill=True)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font("DejaVu", '', 11)
    pdf.ln(5)

    # Bilgi alanı
    fields = {
        "Hasta ID": str(row["hasta_id"]),
        "İsim": str(row["isim"]),
        "Tahmin": str(row["tahmin"]),
        "Güven": str(row["güven"])
    }

    for label, value in fields.items():
        pdf.cell(35, 8, f"{label}:", border=0)
        pdf.cell(60, 8, value, ln=True)

    pdf.ln(5)

    # Görseller yanyana yerleştirilecek
    mask_path = str(row["mask_path"])
    overlay_path = str(row["overlay_path"])

    if os.path.exists(mask_path) or os.path.exists(overlay_path):
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 8, "Segmentasyon Sonuçları", ln=True)

        if os.path.exists(mask_path):
            pdf.image(mask_path, x=30, y=pdf.get_y(), w=65)
        if os.path.exists(overlay_path):
            pdf.image(overlay_path, x=115, y=pdf.get_y(), w=65)

        pdf.ln(70)

    # Açıklama
    pdf.set_font("DejaVu", '', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Bu rapor, yüklenen beyin görüntüsüne yapay zeka destekli sınıflandırma ve segmentasyon işlemleri sonucunda otomatik olarak oluşturulmuştur.")

    os.makedirs("static/pdfs", exist_ok=True)
    output_path = os.path.join("static", "pdfs", f"{hasta_id}_rapor.pdf")
    pdf.output(output_path)

    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
