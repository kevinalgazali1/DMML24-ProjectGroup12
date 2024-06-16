import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Muat data
data = pd.read_csv('test_bersih.csv')
texts = data['text']  # Kolom teks
sentiments = data['sentiment']  # Kolom sentimen

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=42)

# Membuat pipeline untuk vektorisasi dan klasifikasi
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# Melatih model
pipeline.fit(X_train, y_train)

# Menyimpan model
joblib.dump(pipeline, 'sentiment_model.pkl')
