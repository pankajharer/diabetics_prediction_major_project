import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = joblib.load('model.pkl')
sc = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features).reshape(1, -1)
    
    scaled_features = sc.transform(final_features)
    prediction = model.predict(scaled_features)

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
