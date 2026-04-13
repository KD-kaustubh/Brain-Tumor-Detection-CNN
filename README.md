# Brain Tumor Detection (CNN + Flask)

This project detects brain tumors from MRI images using a pre-trained CNN model and a Flask web application.

## Features
- Upload MRI images and run tumor prediction.
- Drag-and-drop upload support.
- Instant image preview before submit.
- Click preview to open full-size image viewer.
- Confidence percentage with confidence level (High/Medium/Low).
- Reset button to clear current state quickly.
- Light/Dark mode toggle (saved in browser).

## Project Structure
```
brain-tumor-detection/
|-- app.py
|-- README.md
|-- requriments.txt
|-- model/
|   `-- brain_tumor_detector.h5
|-- notebook/
|   `-- Brain_Tumor_Detection.ipynb
|-- static/
`-- templates/
    `-- index.html
```

## Tech Stack
- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy

## Setup and Run
1. Create and activate a virtual environment (recommended).

2. Install dependencies:
```bash
pip install -r requriments.txt
```

3. Start the Flask app:
```bash
python app.py
```

4. Open in browser:
```text
http://127.0.0.1:5000
```

## Model Details
- Model file loaded by the app: `model/brain_tumor_detector.h5`
- Input preprocessing in `app.py`:
  - Resize to `128x128`
  - Normalize pixel range to `[0, 1]`
  - Add batch dimension

## Prediction Output
- Label: `Tumor` or `No Tumor`
- Confidence: class-aware score shown as percentage in UI
- Confidence level:
  - High: >= 85%
  - Medium: 60% to 84.99%
  - Low: < 60%

## Notes
- This tool is for educational/research support only and is not a medical diagnosis system.
- If you see model file errors, verify that `model/brain_tumor_detector.h5` exists.
