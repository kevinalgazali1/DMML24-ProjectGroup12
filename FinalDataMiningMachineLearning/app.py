from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Muat model yang telah dilatih sebelumnya
model = joblib.load('sentiment_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        prediction = model.predict([text])[0]  # Asumsi model mengembalikan array
        return render_template('index.html', prediction=prediction, text=text)
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
