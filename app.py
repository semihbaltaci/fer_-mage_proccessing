from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Yuz Ifadesi Tanima API'si Calisiyor!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Lutfen bir gorsel yukleyin"}), 400
    
    file = request.files['file']
    
    # NOT: Görüntü işleme (YOLOv8) modelini ve tahmin kodlarını ileride buraya entegre edeceksin.
    # Şimdilik sadece API'nin başarılı bir şekilde ayağa kalktığını test ediyoruz.
    
    return jsonify({"durum": "Basarili", "mesaj": "Gorsel alindi, model baglantisi bekleniyor."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
