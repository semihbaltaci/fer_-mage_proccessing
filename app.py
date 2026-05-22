from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

try:
    model = YOLO('yolov8n-cls.pt')
    print("Model basariyla yuklendi!")
except Exception as e:
    print(f"Model yuklenirken hata: {e}")

@app.route('/', methods=['GET'])
def home():
    return "FER API Aktif ve Model Hazir!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Gorsel bulunamadi"}), 400
    
    try:
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        results = model(image)
        
        top_class_id = results[0].probs.top1
        predicted_emotion = results[0].names[top_class_id]
        confidence = float(results[0].probs.top1conf)
        
        return jsonify({
            "duygu": predicted_emotion,
            "guven_skoru": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
