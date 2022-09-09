import numpy as np
from flask import Flask, jsonify, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    value = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', prediction_text='{}'.format(value))


if __name__ == "__main__":
    app.run()