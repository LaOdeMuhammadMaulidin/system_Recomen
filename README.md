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
Teknik yang digunakan dalam penyiapan data *(Data Preparation)* yaitu:
- **Handling Imbalanced Data** : Seperti yang telah diketahui sebelumnya bahwa jumlah rating tidak seimbang (imbalance) yang mana sebagian besar user memberikan rating 0 pada buku. Hal ini dapat mengakibatkan model memiliki kinerja yang buruk. Untuk mengatasi hal tersebut, pada proyek ini data dengan rating 0 akan dihapus *(di-drop)*. Walaupun jumlah data saat ini berkurang drastis namun distribusi data menjadi lebih seimbang dan diharapkan memiliki kinerja yang lebih baik.
- **Encoding** : dilakukan untuk menyandikan `User-ID` dan `ISBN` ke dalam indeks integer. Tahapan ini diperlukan karena kedua data tersebut berisi integer yang tidak berurutan (acak) dan gabungan string. Untuk itu perlu diubah ke dalam bentuk indeks.
- **Randomize Dataset** : pengacakan data agar distribusi datanya menjadi random. Pengacakan data bertujuan untuk mengurangi varians dan memastikan bahwa model tetap umum dan *overfit less*. Pengacakan data juga memastikan bahwa data yang digunakan saat validasi merepresentasikan seluruh distribusi data yang ada.
- **Data Standardization** : Pada data rating yang digunakan pada proyek ini berada pada rentang 0 hingga 10. Penerapan standarisasi menjadi rentang 0 hingga 1 dapat mempermudah saat proses training. Hal ini dikarenakan variabel yang diukur pada skala yang berbeda tidak memberikan kontribusi yang sama pada model fitting & fungsi model yang dipelajari dan mungkin berakhir dengan menciptakan bias jika data tidak distandarisasi terlebih dulu.
- **Data Splitting** : dataset dibagi menjadi 2 bagian, yaitu data yang akan digunakan untuk melatih model (sebesar 80%) dan data untuk memvalidasi model (sebesar 20%). Tujuan dari pembagian data uji dan validasi tidak lain adalah untuk proses melatih model serta mengukur kinerja model yang telah didapatkan.

## Modeling
Pada tahap ini, model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. 

Beberapa properti yang digunakan dalam kelas RecommenderNet dan menjadi parameter pada layer embedding untuk menghasilkan model diantaranya:
- `num_users` : jumlah data pengguna
- `num_isbn` : jumlah data buku, dihitung berdasarkan ISBN
- `embedding_size` : ukuran atau dimensi yang digunakan dalam embedding pada data user dan buku

Pertama, kita melakukan proses embedding terhadap data user dan buku. Jumlah user dan buku yang didefinisikan pada `num_users` dan `num_isbn` bertujuan sebagai input untuk membuat vektor embedding keduanya. Sedangkan `embedding_size` menentukan ukuran atau dimensi embedding yang dibuat. Semakin besar nilai dari `embedding_size` akan membuat model semakin akurat, namun jika berlebihan akan mengakibatkan model menjadi overfit. Untuk itu pada proyek ini juga menggunakan `optuna` untuk mencari nilai yang optimal. Selanjutnya, dilakukan operasi perkalian *dot product* antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Model ini juga di-compile dengan fungsi loss binarycrossentropy dan menggunakan Adam sebagai optimizer dengan learning rate sebesar 0.001

Model yang telah dibuat dapat menghasilkan top-10 rekomendasi buku seperti yang ditunjukkan berikut ini.

![Top-10 Book Recommendation](https://github.com/user-attachments/assets/a0527e81-996e-4d7a-89fc-1b410db21ce3)
)

## Evaluation
Pada proyek ini menggunakan metrik RMSE (Root Mean Square Error) untuk mengevaluasi kinerja model yang dihasilkan. RMSE adalah cara standar untuk mengukur kesalahan model dalam memprediksi data kuantitatif [[2](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)]. Root Mean Squared Error (RMSE) mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. Perhitungan RMSE ditunjukkan pada rumus berikut ini.

![RMSE](https://i.postimg.cc/tgjfntZk/RMSE.png)

`RMSE` = nilai root mean square error

`y`  = nilai hasil observasi

`ŷ`  = nilai hasil prediksi

`i`  = urutan data

`n`  = jumlah data

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Berikut ini adalah plot metrik RMSE setelah proses pelatihan model.

![Model Metrics](https://github.com/user-attachments/assets/97b5af90-8f48-445b-b677-1e12d02edfee)
)





  


