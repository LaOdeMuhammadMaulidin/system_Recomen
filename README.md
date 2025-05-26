# Laporan Proyek Machine Learning - La Ode Muhammad Maulidin

## Domain Proyek

Pada saat ini, penyakit stroke menjadi penyakit yang bisa berpotensi menyebabkan kematian. Di seluruh dunia, jumlah kasus stroke terus meningkat, dan memahami faktor-faktor yang mempengaruhi resiko kematian pada pasien stroke adalah langkah penting dalam penanganan dan perawatan yang lebih baik. Stroke adalah tanda-tanda klinis yang terjadi secara cepat atau mendadak berupa defisit fokal (atau global) pada fungsi otak, dengan gejala yang berlangsung selama 24 jam atau lebih atau menyebabkan kematian, tanpa penyebab yang jelas selain penyebab vaskuler (WHO). Menurut data World Health Organization (WHO) diperkirakan 17,5 juta orang meninggal dunia akibat penyakit kardiovaskular dengan 6,7 juta orang meninggal akibat stroke, yaitu urutan kedua tertinggi mengakibatkan kematian setelah penyakit jantung koroner. 

Dengan perkembangan teknologi yang semakin pesat kita dapat dengan mudah mengetahui resiko terkena stroke dengan beberapa gejala yang di alami. salah satu metode tersebut adalah machine learning yang merupakan proses belajar komputer tanpa harus di program secara eksplisit. dengan adanya ML kita dapat mengetahui serta mengklasifikasikan apakah kita beresiko stroke atau tidak
  
[Prediksi Penyakit Stroke Menggunakan Machine Learning Dengan Algoritma Random Forest](https://e-jurnal.pnl.ac.id/infomedia/article/view/5199 ) 

## Business Understanding


### Problem Statements

Beradasarkan latar belakang di atas didapatkan rincian masalah:
- Dengan tingginya angka kematian stroke secara golbal, bagaimana cara mengidentifikasi resiko stroke menggunakan Machine Learning algoritma Random Forest?
- Bagaimana melakukan evaluasi untuk membuat model machine learning yang efisien guna mengidentifikasi resiko stroke?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Memprediksi risiko terkena stroke berdasarkan faktor-faktor kesehatan menggunakan algoritma Machine Learning Random Forest.
- Menganalisis metrik evaluasi model Random Forest dan fitur-fitur penting yang teridentifikasi untuk memastikan model tidak hanya akurat dalam memprediksi risiko   stroke, tetapi juga memberikan wawasan yang relevan dan dapat diinterpretasikan untuk mendukung pengambilan keputusan klinis.


    ### Solution statements
    - melakukan cleaning data dengan menghilangkan missing value yang ada dan menghapus duplikat dari data
    - melakukan splitt data dengan rasio 80% data training dan 20% data testing
    - menggunakan random forest clasifier untuk mengidentifikasi resiko diabetes dengan parameter yang sesuai.
    - melakukan evaluasi dengan mengetahui accuracy, recall, presision dan f1 score serta menampilkan confussion matriks

## Data Understanding

 Dataset yang dogunakan adalah dataset prediksi resiko stroke V2 ini di dapatkan di [Kaggle] https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset-v2 
 ![img](https://github.com/user-attachments/assets/5dbead7e-47c9-4ada-8bac-b4bd7c69aa5d)
  DATASET ini berjumlah 35.000 data dan 19 Kolom.
### Variabel-variabel pada dataset adalah sebagai berikut:
- age : merupakan variabel yang menampilkan umur seseorang.
- gender : merupakan variabel yang menampilkan jenis kelamin seseorang.
- chest pain : merupakan variabel yang menunjukan apakah seseorang mengalami sakit pada dada atau tidak.
- high_blood_pressure:  merupakan variabel apakah tekanan darah tinggi atau tidak
- irregular_heartbeat : merupakan variabel apakah detak jantung tidak normal atau normal
- shortness_of_breath : merupakan variabel apakah bernafas dengan baik atau tidak
- fatigue_weakness    : merupakan variabel apakah sering lelah atau tidak
- dizziness : merupakan variabel apakah sering pusing atau tidak
- swelling_edema : merupakan variabel apakah mengalami pembengkakkan edema atau tidak
- neck_jaw_pain : merupakan variabel apakah mnderita sakit pada rahang atau tidak
- excessive_sweating : merupakan variabel apakah bekeringat berlebih atau tidak
- persistent_cough merupakan variabel apakah menderita batuk atau tidak
- nausea_vomiting: merupakan variabel apakah menderita mual dan mutah atau tidak
- chest_discomfort: merupakan variabel apakah dada merasa nyaman atau tidak
- cold_hands_feet: merupakan variabel apakah tangan dan kaki dingin atau tidak
- snoring_sleep_apnea: merupakan variabel apakah mendengkur saat tidur atau tidak
- anxiety_doom: merupakan variabel apakah mengalami kecemasan berlebih atau tidak
- stroke_risk_percentage: merupakan presentasi dari resiko stroke
- at_risk: merupakan variabel apakah beresiko stroke atau tidak
  **Proses EDA**
  Pada proses Visualisasi untuk melihat perbadingan dataset di peroleh
  - gender : pada kolom ini perbadingan datanya 49,9% untuk perempuan dan 50,1% untuk laki laki
  - chest_pain : pada kolom ini perbadingan datanya 78,3% untuk perempuan dan 21,7% untuk laki laki
  - high_blood_pressure : pada kolom ini perbadingan datanya 65,3% untuk perempuan dan 34,7% untuk laki laki
  - iregullar_heartbeat : pada kolom ini perbadingan datanya 84% untuk perempuan dan 16% untuk laki laki
  - shortness_of_breath : pada kolom ini perbadingan datanya 73,2% untuk perempuan dan 26,8% untuk laki laki
  - fatigue_weakness : pada kolom ini perbadingan datanya 68% untuk perempuan dan 32% untuk laki laki
  - dizziness : pada kolom ini perbadingan datanya 73,5% untuk perempuan dan 26,5% untuk laki laki
  - swelling_edema : pada kolom ini perbadingan datanya 78,3% untuk perempuan dan 21,7% untuk laki laki
  - neck_jaw_pain : pada kolom ini perbadingan datanya 84% untuk perempuan dan 16% untuk laki laki
  - excessive_sweating : pada kolom ini perbadingan datanya 84,8% untuk perempuan dan 15,2% untuk laki laki
  - persistent_cough : pada kolom ini perbadingan datanya 83,3% untuk perempuan dan 16,7% untuk laki laki
  - nausea_vomiting : pada kolom ini perbadingan datanya 85% untuk perempuan dan 15% untuk laki laki
  - chest_discomfort : pada kolom ini perbadingan datanya 78,4% untuk perempuan dan 21,6% untuk laki laki
  - cold_hands_feet : pada kolom ini perbadingan datanya 73% untuk perempuan dan 27% untuk laki laki
  - snoring_sleep_apnea : pada kolom ini perbadingan datanya 77,9% untuk perempuan dan 22,1% untuk laki laki
  - anxiety_doom : pada kolom ini perbadingan datanya 84,9% untuk perempuan dan 15,1% untuk laki laki
  - at_risk : pada kolom ini perbadingan datanya 38,4% untuk perempuan dan 61,6% untuk laki laki
 
  Pada Proses Pengecekkan outlier di peroleh : 
  Ada outlier Pada age
  Jumlah Outlier : 12
  Tidak ada Outlier Pada gender
  Ada outlier Pada chest_pain
  Jumlah Outlier : 4054
  Tidak ada Outlier Pada high_blood_pressure
  Ada outlier Pada irregular_heartbeat
  Jumlah Outlier : 3001
  Tidak ada Outlier Pada shortness_of_breath
  Tidak ada Outlier Pada fatigue_weakness
  Tidak ada Outlier Pada dizziness
  Ada outlier Pada swelling_edema
  Jumlah Outlier : 4068
  Ada outlier Pada neck_jaw_pain
  Jumlah Outlier : 3001
  Ada outlier Pada excessive_sweating
  Jumlah Outlier : 2843
  Ada outlier Pada persistent_cough
  Jumlah Outlier : 3135
  Ada outlier Pada nausea_vomiting
  Jumlah Outlier : 2806
  Ada outlier Pada chest_discomfort
  Jumlah Outlier : 4053
  Tidak ada Outlier Pada cold_hands_feet
  Ada outlier Pada snoring_sleep_apnea
  Jumlah Outlier : 4145
  Ada outlier Pada anxiety_doom
  Jumlah Outlier : 2828
  Tidak ada Outlier Pada at_risk

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


  


