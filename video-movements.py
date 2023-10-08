import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd

# Video dosyasını aç
cap = cv2.VideoCapture('video.mp4')  # Video dosyasının yolunu buraya ekleyin

# Duygu tanıma modelini yükle
emotion_model = load_model('emotion_model.h5')  # Örnek bir duygu tanıma modeli

# Piyano notaları ve frekansları
piano_notes = {
    'Mutlu': ['C4', 'E4', 'G4'],
    'Gerilimli': ['F4', 'A4', 'C5'],
}

# Flüt notaları ve frekansları (örnek)
flute_notes = {
    'Mutlu': ['D5', 'F5', 'A5'],
    'Gerilimli': ['G5', 'B5', 'D6'],
}

# Önceki duygu tahminleri ve parlaklık
prev_emotion = None
prev_brightness = None

# Video için ses çıkışı oluştur
audio_piano = np.array([])
audio_flute = np.array([])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Çerçeve boyutunu yeniden ayarla (isteğe bağlı)
    frame = cv2.resize(frame, (48, 48))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Duygu tahmini yap
    frame = frame / 255.0  # Normalizasyon
    frame = np.expand_dims(frame, axis=0)
    emotion_prediction = emotion_model.predict(frame)
    emotion_label = np.argmax(emotion_prediction)

    # Kare parlaklık seviyesini hesapla
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray_frame)[0]

    # Duygu değişimini kontrol et
    if prev_emotion is not None and prev_emotion != emotion_label:
        # Duygusal tepkiye göre piyano ve flüt notalarını seçin
        if emotion_label == 0:  # Mutlu
            selected_piano_notes = piano_notes['Mutlu']
            selected_flute_notes = flute_notes['Mutlu']
        elif emotion_label == 1:  # Gerilimli
            selected_piano_notes = piano_notes['Gerilimli']
            selected_flute_notes = flute_notes['Gerilimli']
        else:
            selected_piano_notes = []  # Varsayılan olarak hiçbir nota geç
            selected_flute_notes = []

        # Notaları ses sinyaline dönüştür ve çal
        if selected_piano_notes:
            for note in selected_piano_notes:
                frequency = piano_notes[note]
                duration = 1.0  # Her nota 1 saniye sürer
                t = np.linspace(0, duration, int(duration * 44100), False)
                note_signal = np.sin(2 * np.pi * frequency * t)
                audio_piano = np.append(audio_piano, note_signal)

        if selected_flute_notes:
            for note in selected_flute_notes:
                frequency = flute_notes[note]
                duration = 1.0  # Her nota 1 saniye sürer
                t = np.linspace(0, duration, int(duration * 44100), False)
                note_signal = np.sin(2 * np.pi * frequency * t)
                audio_flute = np.append(audio_flute, note_signal)

    prev_emotion = emotion_label

cap.release()

# Ses sinyallerini birleştirin (bu, piyano ve flüt seslerini aynı anda çalacaktır)
combined_audio = audio_piano + audio_flute

# Ses sinyalini bir ses dosyasına kaydet
combined_audio = np.int16(combined_audio / np.max(np.abs(combined_audio)) * 32767)  # Normalize ve tamsayıya dönüştür
cv2.imwrite('output_audio.wav', combined_audio)

# Oynatıcı ile sesi çal
sd.play(combined_audio, 44100)
sd.wait()
