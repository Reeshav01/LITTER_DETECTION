LITTER DETECTION (YOLOv8)

HOW TO RUN
1. Install Python 3.12 (python.org), tick "Add python.exe to PATH"
2. Open a terminal in this folder and run:
   py -3.12 -m pip install ultralytics gradio
3. Start the app:
   py -3.12 app.py
4. Open http://127.0.0.1:7860 in a browser
5. Upload a photo or video and click Detect

FILES
app.py - the web app
best.pt - trained model (YOLOv8n, 10 epochs)
notebook.ipynb - how the model was trained (optional, needs Colab)

MODEL
Dataset: litter-dgwcn (Roboflow Universe, CC BY 4.0)
Validation: precision 0.84, recall 0.87, mAP50 0.92
Limitations: weaker on very small items and extreme close-ups

NOTES
Videos run on CPU, so keep them short.