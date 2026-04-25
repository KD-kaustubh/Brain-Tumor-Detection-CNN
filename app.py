from flask import Flask, render_template, request
import base64
import os
from threading import Lock
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = None
model_lock = Lock()


def get_model():
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = load_model("model/brain_tumor_detector.h5")
    return model

def predict_image(img):
    img = cv2.resize(img, (128,128))
    img = img / 255.0
    img = np.reshape(img, (1,128,128,3))
    
    pred = float(get_model().predict(img, verbose=0)[0][0])
    
    label = "Tumor" if pred > 0.5 else "No Tumor"
    confidence = pred if label == "Tumor" else 1 - pred
    
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    confidence_pct = None
    confidence_level = None
    uploaded_image_data = None
    uploaded_file_name = None
    uploaded_file_size_kb = None
    
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            file_bytes = file.read()
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
            if img is not None:
                uploaded_file_name = file.filename
                uploaded_file_size_kb = round(len(file_bytes) / 1024, 1)
                mime = file.mimetype or "image/jpeg"
                encoded = base64.b64encode(file_bytes).decode("utf-8")
                uploaded_image_data = f"data:{mime};base64,{encoded}"
                result, confidence = predict_image(img)
                confidence_pct = round(confidence * 100, 2)
                if confidence_pct >= 85:
                    confidence_level = "High"
                elif confidence_pct >= 60:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        confidence_pct=confidence_pct,
        confidence_level=confidence_level,
        uploaded_image_data=uploaded_image_data,
        uploaded_file_name=uploaded_file_name,
        uploaded_file_size_kb=uploaded_file_size_kb,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)