# iMosque: Dense Retrieval & Embedding (Kelompok 2)

Proyek ini merupakan bagian dari Tugas Akhir mata kuliah **Advanced Information Retrieval** di **Universitas Nusa Mandiri (Tahun Akademik 2025/2026)**.

**Dosen Pengampu:** Zico Pratama Putra, Lc, MSc., PhD.

---

## Anggota Kelompok 2
* **Joharini** (NIM: 14250008)
* **Farizal Ginanjar** (NIM: 14250014)
* **Feri Stya Heryanto** (NIM: 14250005)

---

## Deskripsi Proyek
iMosque adalah aplikasi pencarian tempat ibadah (masjid/mushola) di Indonesia. Kelompok 2 bertanggung jawab pada implementasi **Dense Retrieval & Embedding**. Kami memproses data tekstual nama masjid menggunakan model-model pembelajaran mendalam (*deep learning*) berbasis **Sentence Transformers** untuk menghasilkan representasi vektor *dense* (embedding) berkualitas tinggi. 

Untuk mendukung pencarian kemiripan semantik (*semantic similarity search*) secara cepat dan efisien pada skala ratusan ribu data, kami membangun indeks pencarian berbasis **FAISS (Facebook AI Similarity Search)**.

---

## Tujuan Eksperimen
1. **Eksperimen Embedding Model**: Membandingkan 4 model Sentence Transformers multibahasa yang berbeda berdasarkan performa representasi semantik nama masjid.
2. **FAISS Indexing**: Membangun indeks berbasis *cosine similarity* menggunakan `faiss.IndexFlatIP` untuk pencarian vektor cepat dan efisien.
3. **Analisis Kinerja**: Mengukur dan membandingkan waktu komputasi pembuatan embedding (*encoding time*) serta ukuran penyimpanan vektor.
4. **Ekspor Data Hasil**: Menyimpan hasil pencarian dari 20 sampel kueri (queries) ke dalam berkas CSV untuk dievaluasi oleh Kelompok 6 (Evaluasi).

---

## Struktur Direktori Proyek
```text
kel-2/
├── Dense_Retrieval_Embedding.ipynb    # Notebook Colab utama (proses embedding & indexing)
├── Folder dense_index/                 # Penyimpanan indeks FAISS & berkas representasi vektor (.npy)
│   ├── BGE_M3.index
│   ├── BGE_M3_embeddings.npy
│   ├── E5_Large.index
│   ├── E5_Large_embeddings.npy
│   ├── MPNet.index
│   ├── MPNet_embeddings.npy
│   ├── MiniLM.index
│   └── MiniLM_embeddings.npy
├── hasil query (4 berkas CSV)/        # Folder hasil ekspor pencarian 20 query untuk 4 model
└── README.md                          # Dokumentasi proyek (berkas ini)
```

---

## Dataset & Pra-pemrosesan Data
* **Dataset Input**: `dataset_preprocessed.csv` berisi **243,831 dokumen** masjid di Indonesia.
* **Fitur Utama**:
  * `name`: Nama masjid (kolom utama yang diproses menjadi embedding).
  * `address`: Alamat lengkap masjid.
  * `provinsi`: Provinsi lokasi masjid.
  * `kabko`: Kabupaten atau Kota lokasi masjid.
  * `kecamatan`: Kecamatan lokasi masjid.
  * `mosque_type`: Kategori tempat ibadah (masjid, mushola, langgar, dll).
* **Tahapan Pra-pemrosesan**:
  * Mengisi nilai kosong (*missing values*).
  * Konversi teks nama masjid ke huruf kecil (*lowercasing*).
  * Pembersihan karakter khusus dan spasi ganda untuk menjaga kualitas embedding.

---

## Model Embedding yang Dibandingkan
Kami melakukan eksperimen menggunakan 4 model Sentence Transformer multibahasa berikut:

| Nama Model | Path Hugging Face | Dimensi Vektor | Karakteristik Utama |
| :--- | :--- | :---: | :--- |
| **BGE-M3** | `BAAI/bge-m3` | 1024 | Akurasi sangat tinggi, mendukung input multibahasa, multi-granularitas, dan representasi dense/sparse/colbert. |
| **E5-Large** | `intfloat/multilingual-e5-large` | 1024 | Model multibahasa kuat yang dilatih secara khusus menggunakan metode pencarian berbasis teks (*text retrieval*). |
| **MPNet** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Menyeimbangkan ukuran model dengan kemampuan pemahaman semantik yang kaya. |
| **MiniLM** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Sangat cepat, ringan, efisien memori, cocok untuk perangkat bersumber daya terbatas. |

---

## Hasil Komputasi & Analisis Performa
Eksperimen dijalankan pada dataset penuh sebanyak **243,831 baris**:

| Model | Dimensi Vektor | Jumlah Dokumen | Waktu Pembuatan Embedding | Ukuran File Embedding (`.npy`) | Ukuran File Indeks FAISS (`.index`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BGE-M3** | 1024 | 243,831 | **116.65 detik** | ~998.7 MB | ~998.7 MB |
| **E5-Large** | 1024 | 243,831 | **106.69 detik** | ~998.7 MB | ~998.7 MB |
| **MPNet** | 768 | 243,831 | **52.12 detik** | ~749.0 MB | ~749.0 MB |
| **MiniLM** | 384 | 243,831 | **35.42 detik** | ~374.5 MB | ~374.5 MB |

### **Kesimpulan Analisis Performa:**
1. **Waktu Komputasi**: `MiniLM` merupakan model tercepat (hanya membutuhkan 110.81 detik), sekitar 5.2x lebih cepat dibandingkan `BGE-M3`.
2. **Efisiensi Ruang**: `MiniLM` menghasilkan indeks FAISS paling kecil (~374.5 MB), yang sangat menghemat memori RAM saat dimuat (*loaded*) ke dalam memori aplikasi.
3. **Akurasi Semantik vs Kecepatan**: Model berdimensi besar seperti `BGE-M3` dan `E5-Large` menawarkan kualitas representasi kata yang lebih kaya namun membutuhkan waktu komputasi serta memori penyimpanan yang jauh lebih besar.

---

## Prasyarat & Instalasi
Sebelum menjalankan notebook, pastikan lingkungan Google Colab telah terinstal pustaka-pustaka berikut:

```bash
pip install sentence-transformers faiss-cpu pandas numpy
```

> [!TIP]
> Jika memiliki GPU Nvidia dengan CUDA terinstal, disarankan untuk menggunakan pustaka FAISS versi GPU untuk mempercepat proses pembuatan indeks:
> `pip install faiss-gpu`

---

## Cara Menjalankan
1. **Persiapkan Dataset**: Unduh dan letakkan file `dataset_preprocessed.csv` di direktori yang sama dengan notebook.
2. **Jalankan Notebook**: Buka berkas `Dense_Retrieval_Embedding.ipynb` melalui Google Colab atau Jupyter Notebook lokal.
3. **Eksekusi Sel Seluruhnya**: Jalankan sel secara berurutan untuk:
   * Membaca dan membersihkan dataset.
   * Melakukan *encoding* nama-nama masjid ke representasi embedding.
   * Membangun indeks FAISS (`IndexFlatIP` dengan normalisasi L2 untuk menghitung *cosine similarity*).
   * Menyimpan hasil ekstraksi indeks (`.index`) dan embedding (`.npy`) ke dalam direktori `Folder dense_index/`.
   * Melakukan pengujian pencarian semantik dengan 20 kueri.
   * Mengekspor hasil pencarian ke direktori `hasil query (4 berkas CSV)/`.

---

## Detail Kueri Uji & Format Output CSV
Sebanyak **20 kueri** diuji menggunakan fungsi `dense_search()` untuk setiap model dengan parameter `top_k=10` (mengembalikan 10 hasil teratas per kueri):

### **Daftar Kueri:**
1. "Masjid Agung"
2. "Masjid Istiqlal"
3. "Masjid Raya Bandung"
4. "Masjid Al Ikhlas"
5. "Masjid Nurul Huda"
6. "Masjid Jami"
7. "Mushola Al Hidayah"
8. "Masjid Al Muttaqin"
9. "Masjid Baiturrahman"
10. "Masjid At Taqwa"
11. "Masjid Al Falah"
12. "Masjid Istiqlal Jakarta"
13. "Masjid Agung Tasikmalaya"
14. "Masjid Agung Menes"
15. "Masjid Nurul Iman"
16. "Islamic Center"
17. "Masjid Al Hikmah"
18. "Masjid Al Amin"
19. "Masjid Darussalam"
20. "Masjid Baitul Makmur"

### **Format Kolom CSV Output:**
Berkas hasil pencarian memiliki format CSV dengan kolom-kolom berikut:
* `name`: Nama masjid hasil pencarian yang cocok.
* `provinsi`: Provinsi lokasi masjid tersebut.
* `kabko`: Kabupaten atau Kota lokasi masjid.
* `kecamatan`: Kecamatan lokasi masjid.
* `address`: Alamat lengkap masjid.
* `mosque_type`: Jenis tempat ibadah.
* `similarity`: Skor kemiripan *Cosine Similarity* (berkisar antara -1 hingga 1, di mana nilai lebih tinggi mendekati 1 menunjukkan tingkat kemiripan semantik yang lebih tinggi).
* `rank`: Peringkat hasil pencarian (1 s.d. 10).
* `query`: Kueri pencarian yang dimasukkan.
* `model`: Nama model embedding yang menghasilkan pencarian ini.

---

## Contoh Implementasi Pencarian Semantik
Berikut adalah contoh potongan kode Python jika ingin memuat indeks FAISS secara langsung dan melakukan pencarian kueri baru di luar notebook:

```python
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Load dataset (untuk pemetaan indeks ke data teks asli)
df = pd.read_csv("dataset_preprocessed.csv")

# 2. Load model embedding dan index FAISS
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name)
index = faiss.read_index("Folder dense_index/BGE_M3.index")

# 3. Definisikan query & encode
query = "masjid raya di jakarta"
query_vector = model.encode([query]).astype(np.float32)

# 4. Normalisasi L2 (untuk Cosine Similarity)
faiss.normalize_L2(query_vector)

# 5. Pencarian similarity
top_k = 5
distances, indices = index.search(query_vector, top_k)

# 6. Tampilkan hasil pencarian
for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), 1):
    mosque_name = df.iloc[idx]['name']
    address = df.iloc[idx]['address']
    print(f"Rank {rank}: {mosque_name} (Similarity: {score:.4f})")
    print(f"Alamat: {address}\n")
```
