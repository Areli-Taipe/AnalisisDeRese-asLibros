# train_model.py

import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Función de limpieza de texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar puntuación
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Eliminar stop words
    return text

# Cargar el dataset
df = pd.read_csv('reviews.csv')
df['review'] = df['review'].apply(clean_text)  # Limpiar reseñas

# Dividir los datos en conjunto de entrenamiento y prueba
X = df['review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizar las reseñas
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Entrenar el modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Guardar el modelo y el vectorizador
joblib.dump(model, 'model/sentiment_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Modelo y vectorizador guardados correctamente.")
