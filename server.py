from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)
scaler = joblib.load('scaler.pkl')

with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)

rnn_model = load_model('rnn_model.h5')  

def preprocess_input(data):
    expected_columns = ['accountAgeDays', 'numItems', 'localTime', 'paymentMethodAgeDays',
                        'paymentMethod_creditcard', 'paymentMethod_paypal', 'paymentMethod_storecredit']
    
    df = pd.DataFrame([data])

    df = pd.get_dummies(df)
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    input_scaled = scaler.transform(df)

    input_array_rnn = np.reshape(input_scaled, (input_scaled.shape[0], 1, input_scaled.shape[1]))

    return input_scaled, input_array_rnn

@app.route('/predict', methods=['POST'])
def predict():
    """
        request format
        {
        "accountAgeDays":1,
        "numItems":1,
        "localTime" :0.00277777777778,
        "paymentMethodAgeDays" : 0,
        "paymentMethod" : "creditcard/paypal"
        }
    """
    data = request.json
    input_array,input_array_rnn = preprocess_input(data)

    lr_prediction = logistic_model.predict(input_array)

    rnn_prediction = rnn_model.predict(input_array_rnn)
    
    
    ensemble_pred = (lr_prediction + rnn_prediction) / 2
    ensemble_pred_binary = int(np.round(ensemble_pred[0][0]))  # Convert to binary (0 or 1)

    response = {
        'result' : ensemble_pred_binary
    }
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
