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
    app.run(host='0.0.0.0', port=5000) 
