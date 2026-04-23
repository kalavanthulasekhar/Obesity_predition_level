from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

app = Flask(__name__)

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

# DB setup
def init_db():
    conn = sqlite3.connect("/tmp/database.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = []

        for key, value in request.form.items():
            if key in encoders:
                val = encoders[key].transform([value])[0]
            else:
                val = float(value)
            input_data.append(val)

        input_array = np.array(input_data).reshape(1, -1)
        scaled = scaler.transform(input_array)

        prediction = model.predict(scaled)[0]

        # Decode result
        result = encoders["NObeyesdad"].inverse_transform([prediction])[0]

        # Save to DB
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO predictions (result) VALUES (?)", (result,))
        conn.commit()
        conn.close()

        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)