# Laporan Proyek Machine Learning - La Ode Muhammad Maulidin

## Domain Proyek

Di era digital saat ini, jumlah informasi yang tersedia secara online terus meningkat secara eksponensial, termasuk dalam industri literasi dan perpustakaan digital. Masyarakat modern semakin terbiasa mengakses buku secara daring melalui platform e-commerce, perpustakaan digital, dan aplikasi baca online. Hal ini menyebabkan tantangan baru bagi pengguna dalam menemukan buku yang relevan atau sesuai dengan preferensi mereka di antara ribuan hingga jutaan pilihan yang tersedia.Salah satu solusi yang dapat diterapkan untuk mengatasi permasalahan tersebut adalah dengan membangun sistem rekomendasi buku. Sistem ini bertujuan untuk menyajikan rekomendasi buku yang bersifat personal, berdasarkan riwayat pembacaan pengguna, penilaian (rating), minat, atau karakteristik buku itu sendiri. Dengan adanya sistem rekomendasi, pengalaman pengguna dalam menemukan bacaan yang sesuai dapat ditingkatkan secara signifikan, serta mendukung peningkatan minat baca masyarakat.

Sistem rekomendasi telah banyak digunakan oleh perusahaan teknologi besar seperti Amazon, Goodreads, dan Google Books. Umumnya, sistem ini dibangun dengan menggunakan algoritma machine learning seperti content-based filtering, collaborative filtering, atau gabungan keduanya dalam bentuk hybrid recommendation system. Salah satu metode yang populer dalam collaborative filtering adalah dengan menggunakan pendekatan berbasis data rating pengguna, yang memprediksi preferensi pengguna baru berdasarkan kesamaan dengan pengguna lain.

  
[Sistem Recomendasi Buku Menggunakan ML](https://ejurnal.umri.ac.id/index.php/coscitech/article/view/5131)) 

## Business Understanding


### Problem Statements

Beradasarkan latar belakang di atas didapatkan rincian masalah:
- pengguna menghadapi kesulitan dalam menemukan buku yang relevan dengan minat dan preferensi pribadi mereka.
- Bagaimana melakukan evaluasi untuk membuat model machine learning yang efisien guna mengidentifikasi resiko stroke?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan sistem rekomendasi buku yang dapat membantu pengguna menemukan bacaan sesuai dengan minat atau riwayat interaksi mereka.
- Mengurangi waktu dan usaha pengguna dalam mencari buku yang sesuai dengan preferensinya.


    ### Solution statements
    - Menganalisis riwayat interaksi pengguna seperti rating atau pencarian buku
    - Menampilkan daftar rekomendasi buku secara otomatis dan dinamis sesuai perilaku pengguna.

## Data Understanding

 ![Book Recommendation Dataset](https://i.postimg.cc/0Q4fcMDB/rsz-bookrecommendationdataset.jpg)

Informasi Dataset:

Jenis | Keterangan
--- | ---
Title | Book Recommendation Dataset
Source | [Kaggle](https://www.kaggle.com/arashnic/book-recommendation-dataset)
Maintainer | [Möbius](https://www.kaggle.com/arashnic)
License | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
Usability | 10.0

Pada Dataset ini terdapat 3 berkas csv diantaranya yaitu `Books.csv` , `Ratings.csv` , dan `Users.csv`

Pada berkas `Books.csv` memuat data-data buku yang terdiri dari 271.360 baris dan memiliki 8 kolom, diantaranya adalah :  

- `ISBN` : berisi kode ISBN dari buku  
- `Book-Title` : berisi judul buku
- `Book-Author` : berisi penulis buku
- `Year-Of-Publication` : tahun terbit buku  
- `Publisher` : penerbit buku  
- `Image-URL-S` : URL menuju gambar buku berukuran kecil
- `Image-URL-M` : URL menuju gambar buku berukuran sedang
- `Image-URL-L` : URL menuju gambar buku berukuran besar

Pada berkas `Ratings.csv` memuat data rating buku yang diberikan oleh pengguna. Data ini memiliki 1.149.780 baris dan memiliki 3 kolom, yaitu :  

 - `User-ID` : berisi ID unik pengguna
 - `ISBN` : berisi kode ISBN buku yang diberi rating oleh pengguna
 - `Book-Rating` : berisi nilai rating yang diberikan oleh pengguna berkisar antara 0-10

![Cuplikan Data Rating](https://i.postimg.cc/wB2m0Qnc/Screenshot-39.png)

## Data Preparation
Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data:
- Meload Dataset ke dalam sebuah Dataframe menggunakan pandas
- menghapus kolom yang tidak digunakan dan kali ini yang di hapus adalah kolom stroke_risk_percentage
- ``` df.info()``` digunakan untuk mengecek tipe kolom pada dataset
- ```df.isna().sum()``` digunakan untuk mengecek apakah ada kolom yg kosong, ternyata pada dataset ini tidak ditemukan missing value. ketika ada missing value maka di atasi dengan   ``` df.dropna(inplace=True)``` 
- ```df.describe()``` digunakan utk mendapatkan info mengenai dataset terhadap nilai rata-rata, median, banyaknya data, nilai Q1 hingga Q3 dan lain-lain
- melakukan pengecekan duplikat dengan ```df.duplicated().sum()```. Pada Dataset ini ditemuka sebanyak 16279 data duplikat dan dilakukan penghapusan duplikat dengan cara ```df.drop_duplicates(inplace=True)```
- Melakukan mapping terhadap kolom diagnosis dari type object ke numerik agar bisa dibaca mesin. Dimana pada kolom age Male  diubah ke nilai 1 female diubah ke nilai 0
- Melakukan pengecekkan distribusi kelas target serta membagi data menjadi data latih dan data test dengan rasio 80 banding 20% serta melihat penyebaran data test dan data latih
![img](https://github.com/user-attachments/assets/27d65e3c-ead1-4371-a470-ce28dcbead0a)

## Modeling
Pada tahap ini dilakukan pembuatan model ML random forest dengan kroteria sebagai berikut : 
![Model Random Forest](![image](https://github.com/user-attachments/assets/dca84a82-8311-4786-8d6c-944e147a683a)

 ### Cara Kerja
Mula mula import library model random forest ```from sklearn.ensemble import RandomForestClassifier``` kemudian import library untuk evaluasi ``` from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score``` kemudian setelah itu baut variabel yang berisi ```RandomForestClassifier``` yang Secara acak mengambil sampel data pelatihan dengan penggantian Proses ini diulang sebanyak jumlah pohon yang ditentukan oleh parameter n_estimators (dalam kasus ini, 100 kali). Setiap sampel bootstrap akan digunakan untuk melatih satu pohon keputusan.Setelah semua pohon (sesuai dengan n_estimators) selesai dibangun, kumpulan pohon ini akan menjadi Random Forest yang terlatih. Setiap pohon akan membuat prediksi kelas untuk sampel tersebut berdasarkan aturan-aturan yang telah dipelajarinya selama pelatihan.Hasil prediksi dari semua pohon akan diagregasikan. Untuk klasifikasi, metode agregasi yang paling umum adalah majority voting: kelas yang paling sering diprediksi oleh semua pohon akan menjadi prediksi akhir untuk sampel tersebut. Hasil prediksi untuk semua sampel dalam test_features akan dikembalikan dalam bentuk array NumPy dan disimpan dalam variabel predict.

### Parameter
Parameter ini menentukan jumlah pohon keputusan dalam forest. Semakin banyak pohon yang dibangun, biasanya kinerja model akan lebih baik dan lebih stabil (kurang rentan terhadap overfitting hingga titik tertentu). Namun, dengan jumlah pohon yang sangat banyak, waktu pelatihan dan prediksi juga akan meningkat, dan manfaat penambahannya mungkin akan berkurang. nilai 100 Ini berarti model Random Forest akan terdiri dari 100 pohon keputusan yang berbeda.

## Evaluation
Pada proyek ini, model yang dikembangkan adalah kasus klasifikasi dan menggunakan metriks akurasi, f1-score, recall dan precision. Berikut hasil pengukuran model yang dipilih yaitu model yang menggunakan algoritma Random Forest
![Metriks Evaluasi](https://github.com/user-attachments/assets/60211e3c-9ab6-48f8-9592-0d10bf9cf8c0)

-Akurasi Akurasi merupakan metrik untuk menghitung persentase dari total data yang diidentifikasi dan dinilai benar. Rumus akurasi sebagai berikut : ``` AKURASI = (TP + TN) / (TP+FP+FN+TN)```
* _True Positive_ (TP):
    Kasus dimana model merupakan data positif yang diprediksi benar. Contohnya, pasien menderita stroke (class 1) dan dari model yang dibuat memprediksi pasien tersebut menderita stroke (class 1).
    * _True Negative_ (TN):
    Kasus dimana model merupakan data negatif yang diprediksi benar. Contohnya, pasien tidak menderita stroke (class 2) dan dari model yang dibuat memprediksi pasien tersebut tidak menderita stroke (class 2).
    * _False Positive_ (FP) - **Type I Error** :
    Kasus dimana model merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita stroke (class 2) tetapi dari model yang telah memprediksi pasien tersebut menderita stroke (class 1).
    * _False Negative_ (FN) - **Type II Error** :
    Kasus dimana model merupakan data negatif namun diprediksi sebagai data positif. Contohnya, pasien tidak menderita stroke (class 2) tetapi dari model yang telah memprediksi pasien tersebut menderita stroke (class 1).
* _Precision_
    _Precision_ merupakan metrik untuk memprediksi benar positif dari keseluruhan hasil yang diprediksi positf. Rumus _precision_ sebagai berikut: ``` PRECISION = TP / (TP+FP)```
  _Recall_
    _Recall_ merupakan metrik untuk memprediksi benar positif dibandingkan dengan keseluruhan data yang benar positif. Rumus _precision_ sebagai berikut: ``` RECALL = TP / (TP+FN)```
  f1-score_
    _f1-score_ merupakan metrik untuk perbandingan rata-rata precision dan recall yang dibobotkan. Rumus _f1-score_ sebagai berikut: ``` F1 SCORE = 2 * (RECALL * PRECISION) / (RECALL + PRECISION) ```
  
- Model yang telah dibangun telah menjawab problem statement karena dapat mengidentifikasi resiko diabetes dengan akurasi yang tinggi serta dapat melakukan evaluasi terhadap model tersebut dengan banyak parameter evaluasi yang digunakan
- Hasil model sangat mencapai target sesuai yang diharapkan dengan bagusnya akurasi model
- Setiap solusi statement yang saya rencanakan berdampak pada model
  **Pertama** : data tidak memiliki missing value dan penghapusan duplikat berdampak Pencegahan Bias dimana Data duplikat dapat memberikan bobot yang tidak semestinya pada sampel yang sama selama pelatihan model. Ini dapat menyebabkan model menjadi bias terhadap sampel duplikat dan kurang mampu menggeneralisasi dengan baik pada data yang unik serta Peningkatan Efisiensi Pelatihan
  **Kedua** : Split data tidak secara langsung mengubah cara kerja algoritma Random Forest, tetapi memungkinkan melatih dan mengevaluasi model secara efektif. Model Random Forest akan dilatih menggunakan 80% data, dan kemampuannya untuk menggeneralisasi akan diukur pada 20% data yang tidak pernah dilihatnya selama pelatihan.
  **Ketiga** : Pemilihan parameter yang sesuai akan secara langsung mempengaruhi kinerja model Random Forest. Parameter yang optimal akan memungkinkan model untuk mempelajari pola risiko diabetes secara efektif dari data pelatihan dan membuat prediksi yang akurat pada data pengujian.
  **Terakhir** : accuracy, recall, presision dan f1 score serta menampilkan confussion matriks tidak mengubah model itu sendiri, tetapi memberikan pemahaman tentang seberapa baik model tersebut bekerja dalam mengidentifikasi risiko diabetes.


  


