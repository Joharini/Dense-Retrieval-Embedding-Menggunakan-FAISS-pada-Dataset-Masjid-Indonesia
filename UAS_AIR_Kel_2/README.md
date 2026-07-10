
# Dense Retrieval & Embedding Menggunakan FAISS

## Advanced Information Retrieval (AIR)

### Kelompok 2 (Joharini, Farizal Ginanjar, Feri Stya Heraynto)

---

# Deskripsi

Project ini bertujuan untuk membangun sistem **Dense Semantic Retrieval**
menggunakan empat model embedding berbasis Transformer dan
FAISS sebagai vector index.

Dataset yang digunakan merupakan dataset masjid Indonesia
(iMosque Dataset).

---

# Tujuan

1. Membandingkan performa empat model embedding.
2. Membangun Dense Retrieval menggunakan FAISS.
3. Melakukan Semantic Search berdasarkan query pengguna.
4. Membandingkan kualitas hasil retrieval setiap model.

---

# Dataset

Jumlah Dokumen : 243,576

Atribut yang digunakan

- name
- address
- provinsi
- kabko
- kecamatan
- mosque_type
- latitude
- longitude

---

# Tahapan Implementasi

## Bagian A

### 1. Import Library

import pandas as pd
import numpy as np
import faiss
import time

from sentence_transformers import SentenceTransformer

Mengimpor library yang digunakan.

### 2. Load Dataset

corpus = pd.read_csv("corpus_clean.csv")

Membaca dataset hasil preprocessing.

### 3. Data Understanding

- Informasi Dataset
- Missing Value
- Statistik Dataset

### 4. Data Cleaning

- Menghapus duplicate
- Mengatasi Missing Value
- Membersihkan Text
- Normalisasi data

### 5. Text Preprocessing

Melakukan preprocessing

- lowercase
- remove punctuation
- remove url
- remove whitespace

### 6. Menyimpan Corpus

Menyimpan hasil preprocessing

corpus_clean.csv

---

# Bagian B

## Dense Retrieval & Embedding

### 1. Load Corpus

Membaca corpus_clean.csv

### 2. Daftar Model Embedding

embedding_models = {
    "BGE-M3": "BAAI/bge-m3",
    "E5-Large": "intfloat/e5-large-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2"
}

Model yang digunakan

- BGE-M3
- E5-Large
- MPNet
- MiniLM

### 3. Inisialisasi Variabel

- loaded_models
- embeddings_dict
- indexes
- embedding_times
- embedding_dimensions

### 4. Generate Embedding

embeddings = model.encode(
    documents,
    show_progress_bar=True,
    convert_to_numpy=True
)

Mengubah seluruh dokumen menjadi vector embedding.

### 5. Ringkasan Embedding

Menampilkan

- Jumlah Vector
- Dimensi Embedding
- Waktu Embedding

### 6. Contoh Embedding

Menampilkan

- Nama Masjid
- Dokumen Setelah Preprocessing
- Contoh Nilai Embedding

### 7. Menyimpan Embedding

np.save(save_path, embeddings)

Format

.npy

### 8. Membangun FAISS Index


faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])

index.add(embeddings)


Menggunakan

IndexFlatIP

dengan Cosine Similarity.

### 9. Menyimpan FAISS Index

faiss.write_index(index, save_path)

Format

.index

### 10. Dense Semantic Search

similarity, indices = indexes[model_name].search(
    query_embedding,
    top_k
)

Membangun fungsi

dense_search()

### 11. Pengujian Single Query

query = "Masjid Istiqlal"

dense_search(
    "MPNet",
    query,
    top_k=10
)

Contoh Query

- Masjid Istiqlal

### 12. Pengujian Multiple Query

- Masjid Raya Baiturrahman
- Masjid Agung Demak
- Masjid Al Jabbar
- Masjid Jami

### 13. Analisis Hasil

results = corpus.iloc[indices[0]].copy()

results["Similarity"] = similarity[0]

Membandingkan

- Similarity
- Waktu Embedding
- Ranking Retrieval

---

# Hasil Eksperimen

Empat model embedding berhasil menghasilkan vector embedding
untuk seluruh dataset dan berhasil membangun FAISS Index.

Hasil pengujian menunjukkan bahwa seluruh model mampu
melakukan semantic retrieval dengan baik.

Model yang digunakan

| Model | Dimensi |
|--------|---------|
| BGE-M3 | 1024 |
| E5-Large | 1024 |
| MPNet | 768 |
| MiniLM | 384 |

---

# Kesimpulan

- Seluruh model berhasil membuat embedding.
- Seluruh model berhasil membangun FAISS Index.
- Seluruh model berhasil melakukan Dense Retrieval.
- MPNet memberikan similarity paling tinggi pada eksperimen ini.
- MiniLM memiliki waktu embedding paling cepat.

---

# Teknologi

- Python
- Google Colab
- SentenceTransformers
- FAISS
- Pandas
- NumPy

---

# Repository

Project dibuat sebagai tugas mata kuliah
Advanced Information Retrieval.

