import pickle
from flask import Flask
from flask import request
from flask import jsonify
from waitress import serve
from sklearn.ensemble import RandomForestClassifier


input_file = 'model_rf.bin'

with open(input_file,'rb') as f_in:
    dv,model = pickle.load(f_in)

app = Flask(__name__)


@app.route('/predict/',methods=['POST'])

def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    X_Dm = rf

    y_pred = model.predict_proba(X)[0,1]
    LiverCirrhosis = float(y_pred) >= 0.5


    result = {
        'LiverCirrhosis_probability': float(y_pred),
        'LiverCirrhosis': bool(LiverCirrhosis)
    }

    return jsonify(result)
