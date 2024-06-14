import cv2
import numpy as np

# Path ke file Haar Cascade
cascade_path = 'haarcascade_frontalface_default.xml'

# Muat Haar Cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Fungsi untuk mendeteksi warna kulit pada gambar
def skin_detection(image):
    # Konversi gambar ke ruang warna YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Definisi batasan warna kulit dalam ruang YCrCb
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    # Masking untuk mendapatkan area kulit
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Aplikasikan mask ke gambar asli
    skin = cv2.bitwise_and(image, image, mask=mask)

    return skin

def skin(image_path):
    # Muat gambar
    image = cv2.imread(image_path)

    # Ubah gambar ke dalam skala abu-abu
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Jika wajah ditemukan, ambil koordinat wajah pertama
    if len(faces) > 0:
        x, y, w, h = faces[0]

        # Gambar persegi di sekitar wajah pertama yang terdeteksi
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Potong bagian wajah dari gambar asli
        face_image = image[y:y+h, x:x+w]

        # Deteksi warna kulit pada bagian wajah
        skin_detected = skin_detection(face_image)

        # Hitung rata-rata warna kulit dalam representasi BGR
        avg_color_per_row = np.average(skin_detected, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)

        # Konversi dari BGR ke RGB
        avg_color_rgb = avg_color[::-1].astype(int)  # BGR ke RGB

        return avg_color_rgb

def eye(image_path):
    # Muat gambar
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lakukan deteksi mata
    eyes = eye_cascade.detectMultiScale(gray)

    # Jika ada mata yang terdeteksi, ambil mata pertama saja
    if len(eyes) > 0:
        (x, y, w, h) = eyes[0]  # Ambil koordinat kotak untuk mata pertama
        eye_roi = image[y:y+h, x:x+w]  # Ambil ROI (Region of Interest) mata

        # Ubah gambar mata menjadi grayscale
        eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # Thresholding gambar mata
        _, eye_thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY)

        # Temukan kontur dalam gambar mata
        contours, _ = cv2.findContours(eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ambil kontur bola mata (kontur terbesar)
        if len(contours) > 0:
            eye_contour = max(contours, key=cv2.contourArea)

            # Hitung lingkaran yang melingkupi kontur bola mata
            ((cx, cy), radius) = cv2.minEnclosingCircle(eye_contour)
            center = (int(cx), int(cy))
            radius = int(radius)

            # Gambar lingkaran pada bola mata
            cv2.circle(eye_roi, center, radius, (0, 255, 0), 2)

            # Hitung rata-rata warna dalam lingkaran melingkupi bola mata
            eye_color_roi = eye_roi[int(cy)-radius:int(cy)+radius, int(cx)-radius:int(cx)+radius]
            eye_color_rgb = np.mean(eye_color_roi, axis=(0, 1)).astype(int)
    
    return eye_color_rgb