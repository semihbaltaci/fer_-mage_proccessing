from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Tüm dış isteklere (GitHub Pages dahil) izin verir

@app.route('/', methods=['GET'])
def home():
    return "Yuz Ifadesi Tanima API'si Calisiyor!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Lutfen bir gorsel yukleyin"}), 400
    
    file = request.files['file']
    return jsonify({"durum": "Basarili", "mesaj": "Gorsel alindi, model baglantisi bekleniyor."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Modelini hafızaya yükle (yolov8n-cls.pt dosyasının app.py ile aynı klasörde olduğundan emin ol!)
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
        # 1. Gelen dosyayı oku ve resme çevir
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Modeli çalıştır ve tahmin yap
        results = model(image)
        
        # 3. En yüksek ihtimalli sonucu al (Sınıflandırma modeli olduğu varsayımıyla)
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
    app.run(host='0.0.0.0', port=5000)) 
