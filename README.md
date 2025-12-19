# Learn Computer Vision

Repositori pembelajaran Computer Vision yang berfokus pada implementasi praktis menggunakan framework deep learning terkini. Materi pembelajaran dirancang untuk pemula hingga menengah yang ingin menguasai konsep dan aplikasi computer vision, khususnya dalam deteksi objek dan analisis wajah.

## 📋 Daftar Isi

- [Fitur Utama](#fitur-utama)
- [Prasyarat](#prasyarat)
- [Setup Lingkungan](#setup-lingkungan)
- [Struktur Kursus](#struktur-kursus)
- [Cara Menggunakan](#cara-menggunakan)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Kontribusi](#kontribusi)

## ✨ Fitur Utama

- **Setup Lingkungan Deep Learning**: Panduan lengkap instalasi TensorFlow dengan dukungan GPU
- **Deteksi Emosi Real-time**: Implementasi YOLOv5 untuk mendeteksi emosi pada wajah manusia
- **Model Pra-latih**: Menggunakan model YOLOv5 yang sudah terlatih untuk akselerasi pembelajaran
- **Jupyter Notebook**: Semua materi tersedia dalam format notebook interaktif untuk pembelajaran berkelanjutan
- **Simpan dan Muat Model**: Fungsi untuk menyimpan dan memuat model terlatih

## 🔧 Prasyarat

Sebelum memulai, pastikan sistem Anda memenuhi persyaratan berikut:

- **Python 3.8 atau lebih tinggi**
- **pip** (Python package manager)
- **GPU NVIDIA** (opsional, tetapi disarankan untuk performa optimal)
- **CUDA Toolkit** dan **cuDNN** (jika menggunakan GPU)
- **Jupyter Notebook** atau **JupyterLab**

### Verifikasi Instalasi Python

```bash
python --version
pip --version
```

## 🚀 Setup Lingkungan

### 1. Clone Repositori

```bash
git clone https://github.com/ShiroKuro017/Learn-Computer-Vision.git
cd Learn-Computer-Vision
```

### 2. Buat Virtual Environment

```bash
# Menggunakan venv
python -m venv venv

# Aktivasi virtual environment
# Untuk Windows:
venv\Scripts\activate
# Untuk macOS/Linux:
source venv/bin/activate
```

### 3. Instalasi Dependencies

Buka notebook `Install TensorFlow w GPU.ipynb` dan ikuti langkah-langkah instalasi untuk mengatur TensorFlow dengan dukungan GPU atau CPU sesuai kebutuhan.

Alternatif, instal dependencies secara manual:

```bash
pip install tensorflow
pip install torch torchvision torchaudio
pip install yolov5
pip install jupyter
pip install opencv-python
pip install numpy pandas matplotlib
```

### 4. Verifikasi Instalasi

Jalankan snippet berikut untuk memverifikasi TensorFlow terdeteksi GPU:

```python
import tensorflow as tf
print("GPU Available:", tf.test.is_built_with_cuda())
print("Physical GPUs:", len(tf.config.list_physical_devices('GPU')))
```

## 📚 Struktur Kursus

Repositori ini terstruktur dalam modul-modul pembelajaran berikut:

### Modul 1: Persiapan Lingkungan
**File**: `Install TensorFlow w GPU.ipynb`

Materi pembelajaran mencakup:
- Instalasi TensorFlow dengan dukungan GPU
- Konfigurasi CUDA dan cuDNN
- Verifikasi lingkungan development
- Troubleshooting masalah instalasi umum

### Modul 2: Deteksi Emosi dengan YOLOv5
**File**: `Emotion-Detection.ipynb`

Materi pembelajaran mencakup:
- Pengenalan arsitektur YOLOv5
- Deteksi wajah real-time
- Ekstraksi fitur emosi dari wajah
- Implementasi pipeline lengkap deteksi emosi
- Fungsi simpan dan muat model terlatih
- Optimisasi performa pada berbagai hardware

### Folder Models
**Direktori**: `models/`

Penyimpanan model-model terlatih yang dapat digunakan untuk inference atau fine-tuning lebih lanjut.

### Model Pra-latih
**File**: `yolov5s.pt`

Model YOLOv5 small yang sudah terlatih pada dataset COCO untuk deteksi objek dan wajah. File ini digunakan dalam notebook Emotion-Detection.

## 💻 Cara Menggunakan

### Menjalankan Notebook

1. **Aktifkan Virtual Environment**:
   ```bash
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

3. **Navigasi ke Notebook yang Ingin Dipelajari**:
   - Mulai dengan `Install TensorFlow w GPU.ipynb` untuk setup
   - Lanjutkan dengan `Emotion-Detection.ipynb` untuk praktek deteksi emosi

4. **Jalankan Cell-Cell secara Berurutan**:
   - Tekan `Shift + Enter` untuk menjalankan cell
   - Perhatikan output dan hasil visualisasi

### Contoh Penggunaan Dasar

Setelah menyelesaikan setup, Anda dapat menggunakan model untuk deteksi emosi:

```python
# Import dependencies
import yolov5
import cv2

# Load model
model = yolov5.load('yolov5s.pt')

# Jalankan deteksi pada gambar
results = model('path/to/image.jpg')

# Tampilkan hasil
results.show()
```

## 🛠️ Teknologi yang Digunakan

| Teknologi | Versi | Kegunaan |
|-----------|-------|----------|
| Python | 3.8+ | Bahasa pemrograman utama |
| TensorFlow | 2.x | Framework deep learning |
| PyTorch | Latest | Alternatif framework deep learning |
| YOLOv5 | Latest | Arsitektur deteksi objek |
| OpenCV | 4.x | Pemrosesan gambar dan video |
| Jupyter | Latest | Environment notebook interaktif |
| NumPy | Latest | Komputasi numerik |
| Pandas | Latest | Manipulasi data |
| Matplotlib | Latest | Visualisasi data |

## 🎓 Learning Path yang Disarankan

Untuk hasil pembelajaran optimal, ikuti urutan berikut:

1. **Minggu 1**: Setup lingkungan dan verifikasi instalasi
   - Jalankan `Install TensorFlow w GPU.ipynb`
   - Pastikan semua dependencies terinstal dengan baik
   
2. **Minggu 2-3**: Pemahaman konsep YOLOv5
   - Pelajari arsitektur dan cara kerja YOLOv5
   - Pahami konsep deteksi objek real-time
   
3. **Minggu 4-5**: Implementasi Emotion Detection
   - Jalankan `Emotion-Detection.ipynb` secara menyeluruh
   - Eksperimen dengan berbagai input gambar dan video
   
4. **Minggu 6+**: Eksplorasi dan Pengembangan Lanjut
   - Fine-tune model untuk use case spesifik
   - Integrasikan ke aplikasi real-world
   - Eksperimen dengan dataset custom

## 📝 Catatan Penting

- **GPU vs CPU**: Proses training dan inference jauh lebih cepat dengan GPU. Jika tidak memiliki GPU NVIDIA, gunakan Google Colab untuk akses GPU gratis.
- **Model Size**: File `yolov5s.pt` berukuran beberapa ratus MB. Pastikan memiliki space storage yang cukup.
- **Dokumentasi**: Setiap notebook berisi dokumentasi inline dan penjelasan code untuk memudahkan pembelajaran.
- **Compatibility**: Verifikasi versi library Anda kompatibel dengan versi yang digunakan dalam notebook.

## 🐛 Troubleshooting

### CUDA Not Available
```bash
# Pastikan CUDA Toolkit terinstal
nvidia-smi

# Update driver GPU
# (Instruksi spesifik tergantung OS dan GPU)
```

### Import Error
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Memory Error Saat Inference
- Gunakan model yang lebih kecil (yolov5n.pt alih-alih yolov5l.pt)
- Kurangi ukuran input gambar
- Gunakan batch processing yang lebih kecil

## 📂 Struktur Direktori

```
Learn-Computer-Vision/
├── Install TensorFlow w GPU.ipynb      # Modul setup lingkungan
├── Emotion-Detection.ipynb             # Modul deteksi emosi
├── yolov5s.pt                          # Model pra-latih YOLOv5
├── models/                             # Folder penyimpanan model custom
└── README.md                           # File dokumentasi ini
```

## 🤝 Kontribusi

Kontribusi sangat diterima! Berikut cara berkontribusi:

1. **Fork** repositori ini
2. Buat **branch** fitur baru (`git checkout -b feature/AmazingFeature`)
3. **Commit** perubahan Anda (`git commit -m 'Add some AmazingFeature'`)
4. **Push** ke branch (`git push origin feature/AmazingFeature`)
5. Buka **Pull Request**

### Pedoman Kontribusi

- Pastikan code mengikuti style konsisten
- Tambahkan dokumentasi untuk fitur baru
- Test sebelum submit pull request
- Tulis commit message yang deskriptif

## 📄 Lisensi

Repositori ini tersedia di bawah lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.

## 📞 Dukungan dan Pertanyaan

Jika mengalami kesulitan atau memiliki pertanyaan:

- **Issues**: Buka issue baru di GitHub dengan detail lengkap
- **Discussions**: Gunakan discussions untuk pertanyaan umum
- **Email**: Hubungi maintainer melalui GitHub profile

## 🔗 Referensi Tambahan

- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Tutorial](https://docs.opencv.org/)
- [Computer Vision Papers](https://arxiv.org/)

---

**Last Updated**: Desember 2025

Selamat belajar! Jika repositori ini membantu, jangan lupa berikan ⭐ star.
