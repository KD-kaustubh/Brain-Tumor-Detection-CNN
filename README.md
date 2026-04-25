# Brain Tumor Detection (CNN + Flask)

A deep learning web application that predicts whether an uploaded brain MRI image shows Tumor or No Tumor using a trained CNN model.

## Live Demo

https://brain-tumor-detection-cnn-jz82.onrender.com

## Features

- MRI image upload with drag-and-drop support
- Instant image preview before prediction
- Click-to-open full-size image viewer
- Prediction result: Tumor or No Tumor
- Confidence score shown as percentage
- Confidence level labels: High, Medium, Low
- Reset action for quick re-testing
- Light/Dark mode toggle

## Tech Stack

- Python
- Flask
- TensorFlow / Keras
- OpenCV (headless)
- NumPy
- Gunicorn (production server)

## Project Structure

```text
brain-tumor-detection/
|-- app.py
|-- Procfile
|-- runtime.txt
|-- requirements.txt
|-- model/
|   `-- brain_tumor_detector.h5
|-- notebook/
|   `-- Brain_Tumor_Detection.ipynb
|-- static/
`-- templates/
    `-- index.html
```

## Local Setup

1. Clone the repository and enter the project folder.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask app:

```bash
python app.py
```

5. Open in browser:

```text
http://127.0.0.1:5000
```

## Model Inference Pipeline

1. Read uploaded image
2. Resize to 128 x 128
3. Normalize pixel values to [0, 1]
4. Expand dimensions for batch input
5. Run CNN prediction
6. Return class label and confidence

## Deployment (Render)

This project is configured for Render with:

- Procfile start command:
  - gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 2 --timeout 300
- Python version pin:
  - runtime.txt -> python-3.12.10
- Build command:
  - pip install -r requirements.txt

## Notes

- This project is for educational and research support.
- It is not a replacement for professional medical diagnosis.
- Ensure model file exists at model/brain_tumor_detector.h5.
