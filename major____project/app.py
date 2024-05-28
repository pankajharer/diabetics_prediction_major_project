import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = joblib.load('model.pkl')

dataset = pd.read_csv('diabetes-dataset.csv')
dataset_X = dataset.iloc[:, [1, 2, 5, 7]].values

sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))

