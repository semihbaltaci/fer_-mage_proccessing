import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import base64

app = Flask(__name__)
CORS(app)

def get_latest_model(runs_dir="runs/classify", base_model="yolov8n-cls.pt"):
    if not os.path.exists(runs_dir):
        return base_model
    train_folders = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) 
                     if d.startswith('train') and os.path.isdir(os.path.join(runs_dir, d))]
    if not train_folders:
        return base_model
    train_folders.sort(key=os.path.getctime, reverse=True)
    for folder in train_folders:
        best_pt = os.path.join(folder, "weights", "best.pt")
        last_pt = os.path.join(folder, "weights", "last.pt")
        if os.path.exists(best_pt):
            return best_pt
        elif os.path.exists(last_pt):
            return last_pt
    return base_model

model_path = get_latest_model()
try:
    print(f"Model yukleniyor: {model_path}")
    model = YOLO(model_path)
except Exception as e:
    print(f"Model yuklenirken hata: {e}")
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def process_frame(frame):
    if model is None:
        return frame
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        results = model.predict(source=face_img, imgsz=48, verbose=False)
        probs = results[0].probs
        best_class_index = probs.top1
        emotion = results[0].names[best_class_index]
        
        text = f"{emotion}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
        cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
    return frame

@app.route('/')
def index():
    return jsonify({"mesaj": "API Calisiyor"})

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    if model is None:
        return jsonify({'faces': []})
        
    file = request.files.get('frame')
    if not file:
        return jsonify({'faces': []})
        
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'faces': []})
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    
    results_list = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        results = model.predict(source=face_img, imgsz=48, verbose=False)
        probs = results[0].probs
        best_class_index = probs.top1
        emotion = results[0].names[best_class_index]
        
        results_list.append({
            'box': [int(x), int(y), int(w), int(h)],
            'emotion': emotion
        })
        
    return jsonify({'faces': results_list})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadi'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya secilmedi'})
    
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
         return jsonify({'error': 'Gecersiz resim formati'})
         
    frame = process_frame(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'image': img_base64})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
