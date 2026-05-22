import cv2
import os
from ultralytics import YOLO
import numpy as np
from collections import deque

def get_latest_model(runs_dir="runs/classify", base_model="yolov8n-cls.pt"):
    """En son eğitilmiş modeli bulur"""
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
        
        # Öncelikli olarak en iyi başarıma sahip modeli (best.pt) alıyoruz
        if os.path.exists(best_pt):
            return best_pt
        elif os.path.exists(last_pt):
            return last_pt
            
    return base_model

def main():
    # 1. Eğitilmiş en iyi modeli yükle
    model_path = get_latest_model()
    print(f"Kullanılan model: {model_path}")
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    # 2. Yüz algılama için OpenCV'nin hazır Haar Cascade kütüphanesini kullanıyoruz
    # (YOLOv8-cls tüm resmi sınıflandırır. Doğru yapmak için sadece yüzü yakalayıp modele vermeliyiz)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Kamerayı başlat (0 numaralı indeks genellikle bilgisayarın ana kamerasıdır)
    cap = cv2.VideoCapture(0)
    
    # Temporal Smoothing (Zaman Serisi Yumuşatması) için kuyruk
    # Son 3 tahmini saklayarak ani değişimleri engelleriz (Daha hızlı tepki için 3'e düşürüldü)
    emotion_history = deque(maxlen=3)
    
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return

    print("\n" + "="*50)
    print("KAMERA AÇILDI! Çıkmak için klavyeden 'q' tuşuna basın.")
    print("="*50 + "\n")

    while True:
        # Kameradan bir kare (frame) al
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı.")
            break

        # Görüntüyü gri tona çevir (Haar Cascade gri tonda çok daha hızlı ve iyi çalışır)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Görüntüdeki yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        # Tespit edilen her bir yüz için:
        for (x, y, w, h) in faces:
            # 1. Yüzü asıl görüntüden kes
            face_img = frame[y:y+h, x:x+w]
            
            # CLAHE Uygulama (Kontrast artırarak mikro ifadeleri netleştirir)
            if face_img.size > 0:
                # Gri tona çevir
                face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                # CLAHE objesi oluştur
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                face_gray_clahe = clahe.apply(face_gray)
                # Tekrar BGR'ye çevir (YOLO BGR/RGB bekler)
                face_img = cv2.cvtColor(face_gray_clahe, cv2.COLOR_GRAY2BGR)
            
            # 2. YOLO'ya modeli tahmin ettir (Verbose=False yaparak terminalin yazıyla dolmasını engelliyoruz)
            results = model.predict(source=face_img, imgsz=48, verbose=False)
            
            # 3. Sonucu al
            probs = results[0].probs
            best_class_index = probs.top1
            emotion = results[0].names[best_class_index]
            confidence = probs.top1conf.item() * 100

            # Temporal Smoothing Uygula
            emotion_history.append(emotion)
            
            # Ağırlıklı Oylama (Angry ve Sad duygularına pozitif ayrımcılık)
            emotion_counts = {}
            for e in emotion_history:
                # Angry ve Sad için ağırlık 1.5, diğerleri için 1.0
                weight = 1.5 if e in ['angry', 'sad'] else 1.0
                emotion_counts[e] = emotion_counts.get(e, 0) + weight
            
            # En yüksek ağırlığa sahip duyguyu seç
            smoothed_emotion = max(emotion_counts, key=emotion_counts.get)

            # 4. Yüzün etrafına bir dikdörtgen çiz ve tahmin edilen duyguyu yaz
            color = (0, 255, 0) # Yeşil Klasik Renk
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Yazı metni (Örn: "happy: 95.5%")
            text = f"{smoothed_emotion} (%{confidence:.1f})"
            
            # Yazının daha okunabilir olması için arka plan siyah bant çizimi
            cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
            cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 5. Ekrana çizilmiş kareleri göster
        cv2.imshow("Canli Yuz Ifadesi Tanima", frame)

        # Klavyeden 'q' tuşuna basılıp basılmadığını kontrol et
        # cv2.waitKey(1) frame'in ekranda renderlanması için gerekli süredir (1 milisaniye)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Döngü bittiğinde kamerayı serbest bırak ve tüm pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
