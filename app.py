# app.py

from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_vec = vectorizer.transform([review])  # Vectorizar la reseña
    prediction = model.predict(review_vec)  # Hacer la predicción
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
