import pandas as pd  # Mengimpor library pandas untuk manipulasi data
from sklearn.model_selection import train_test_split  # Mengimpor fungsi untuk membagi data menjadi data latih dan data uji
from sklearn.feature_extraction.text import CountVectorizer  # Mengimpor CountVectorizer untuk mengubah teks menjadi vektor
from sklearn.linear_model import LogisticRegression  # Mengimpor model Logistic Regression untuk klasifikasi
from sklearn.pipeline import Pipeline  # Mengimpor Pipeline untuk memudahkan proses transformasi dan pelatihan model
import joblib  # Mengimpor joblib untuk menyimpan model yang telah dilatih

# Muat data dari file CSV
data = pd.read_csv('test_bersih.csv')  # Membaca data dari file CSV
texts = data['text']  # Mengambil kolom teks dari data
sentiments = data['sentiment']  # Mengambil kolom sentimen dari data

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=42)  
# Memisahkan data menjadi 80% latih dan 20% uji

# Membuat pipeline untuk vektorisasi dan klasifikasi
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  # Langkah pertama dalam pipeline: mengubah teks menjadi vektor
    ('classifier', LogisticRegression())  # Langkah kedua dalam pipeline: melakukan klasifikasi menggunakan Logistic Regression
])

# Melatih model menggunakan data latih
pipeline.fit(X_train, y_train)  # Melatih model dengan data latih

# Menyimpan model yang telah dilatih
joblib.dump(pipeline, 'sentiment_model.pkl')  # Menyimpan model ke dalam file 'sentiment_model.pkl'
