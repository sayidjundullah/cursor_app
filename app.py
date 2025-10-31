import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import threading
import time
from pynput.mouse import Controller, Button
import numpy as np

# --- 1. SETUP GLOBAL ---
mouse = Controller()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Variabel EMA Filter (Dipastikan Global agar bisa diakses thread)
prev_x, prev_y = 0, 0
# Alpha: Faktor penghalusan. Nilai kecil (0.1 - 0.3) = lebih halus/lambat.
# Jika Anda ingin lebih halus lagi, coba alpha = 0.15.
alpha = 0.2

# PENTING: Mendapatkan Resolusi Layar menggunakan Tkinter
temp_root = tk.Tk()
screen_width = temp_root.winfo_screenwidth()
screen_height = temp_root.winfo_screenheight()
temp_root.destroy() 

# Variabel Global untuk Komunikasi antar Thread
is_running = False
hand_data = None


# --- 2. LOGIKA DETEKSI TANGAN ---

def calculate_distance(point1, point2):
    """Menghitung jarak Euclidean antara dua landmark."""
    # Menghitung jarak 3D karena landmark MediaPipe menyediakan z-coordinate
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def check_click_gesture(landmarks):
    """
    Menentukan apakah gestur 'klik kiri' terdeteksi.
    Kriteria: Jarak antara ujung Jari Telunjuk (8) dan Ibu Jari (4) sangat kecil.
    """
    if not landmarks:
        return False
        
    # Landmark 4: Ujung Ibu Jari
    # Landmark 8: Ujung Jari Telunjuk
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    distance = calculate_distance(thumb_tip, index_tip)
    
    # Ambang batas (threshold) eksperimental untuk klik
    # Ambang batas ini mungkin perlu disesuaikan (coba 0.040 - 0.050)
    if distance < 0.045: 
        return True
    return False

def video_processing_loop():
    """
    Fungsi ini berjalan di background thread.
    Mengambil frame, mendeteksi tangan, dan menggerakkan kursor.
    """
    # Pastikan semua variabel yang dimodifikasi atau diakses ada di sini
    global is_running, hand_data, screen_width, screen_height, prev_x, prev_y, alpha
    
    # Menggunakan index kamera 0 adalah lebih umum. Jika 1 tidak berfungsi, coba 0.
    cap = cv2.VideoCapture(1) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    
    while is_running:
        success, img = cap.read()
        if not success:
            break
            
        img = cv2.flip(img, 1) # Mirroring untuk tampilan alami
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Ambil landmark untuk TANGAN PERTAMA (yang dideteksi)
                hand_data = hand_landmarks 
                
                # Kursor dikontrol oleh posisi ujung jari telunjuk (Landmark 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # 1. Penskalaan Koordinat Mentah (Raw) dari Kamera (0-1) ke Layar OS (Pixel)
                raw_x = int(index_finger_tip.x * screen_width)
                raw_y = int(index_finger_tip.y * screen_height)
                
                # --- 2. Penerapan EMA Filter untuk Penghalusan ---
                if prev_x == 0 and prev_y == 0:
                    # Inisialisasi awal pada frame pertama
                    prev_x, prev_y = raw_x, raw_y
                
                # Rumus EMA: Smoothed = alpha * Raw + (1 - alpha) * Previous_Smoothed
                smoothed_x = int(alpha * raw_x + (1 - alpha) * prev_x)
                smoothed_y = int(alpha * raw_y + (1 - alpha) * prev_y)

                # 3. Pindahkan Kursor ke Posisi yang Sudah Dihaluskan
                # INI ADALAH POSISI KURSOR AKHIR YANG DITERAPKAN KE OS
                mouse.position = (smoothed_x, smoothed_y)
                
                # 4. Simpan Posisi yang Sudah Dihaluskan untuk Iterasi Berikutnya
                prev_x, prev_y = smoothed_x, smoothed_y
                
                # 5. Cek Gestur Klik (menggunakan hand_landmarks mentah)
                if check_click_gesture(hand_landmarks):
                    mouse.click(Button.left, 1) # Lakukan Klik Kiri
                    time.sleep(0.3) # Jeda untuk mencegah multiple click
                
                # Gambar landmark (opsional, jika Anda ingin debugging visual)
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            hand_data = None
            
    cap.release()
    cv2.destroyAllWindows()
    is_running = False 

# --- 3. IMPLEMENTASI TKINTER UI ---

class CursorControllerApp:
    def __init__(self, master):
        self.master = master
        master.title("Hand Gesture Cursor Controller (by MediaPipe + Tkinter)")
        master.geometry("400x200")
        
        self.is_active = False
        self.worker_thread = None

        # --- Komponen UI ---
        self.status_label = tk.Label(master, text="Status: IDLE", fg="red", font=('Arial', 14, 'bold'))
        self.status_label.pack(pady=10)

        self.btn_start = tk.Button(master, text="START CONTROL", command=self.start_control, bg="green", fg="white", font=('Arial', 12))
        self.btn_start.pack(pady=5, padx=20, fill='x')

        self.btn_stop = tk.Button(master, text="STOP CONTROL", command=self.stop_control, bg="red", fg="white", font=('Arial', 12), state=tk.DISABLED)
        self.btn_stop.pack(pady=5, padx=20, fill='x')

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def start_control(self):
        """Memulai thread video processing."""
        global is_running
        if not is_running:
            try:
                is_running = True
                self.worker_thread = threading.Thread(target=video_processing_loop)
                self.worker_thread.start()
                
                self.is_active = True
                self.update_ui_status()
                messagebox.showinfo("INFO", "Sistem Kontrol Dimulai! Gerakan Anda kini mengontrol kursor.")
                
            except Exception as e:
                messagebox.showerror("ERROR", f"Gagal memulai kontrol: {e}")
                is_running = False

    def stop_control(self):
        """Menghentikan thread video processing."""
        global is_running
        if is_running:
            is_running = False
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1) 
            
            self.is_active = False
            self.update_ui_status()
            messagebox.showinfo("INFO", "Sistem Kontrol Dihentikan.")

    def update_ui_status(self):
        """Memperbarui label status di UI."""
        if self.is_active and is_running:
            self.status_label.config(text="Status: ACTIVE (Mengontrol Kursor)", fg="green")
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Status: IDLE (Siap Memulai)", fg="red")
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def on_closing(self):
        """Menangani penutupan aplikasi."""
        self.stop_control() 
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CursorControllerApp(root)
    root.mainloop()
