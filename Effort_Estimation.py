from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import StandardScaler
import joblib
import json

scaler = StandardScaler()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the pre-trained model
modellinear = joblib.load('trained_model.pkl')
svr=joblib.load('svr.pkl')
dt=joblib.load('DecisionTree.pkl')
mlp=joblib.load('mlp.pkl')
smo=joblib.load('smo_polynomial.pkl')
scaler=joblib.load('fitted_scaler.pkl')

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json(force=True)
    noa = int(data['NOCA'])
    nem = int(data['NOCM'])
    nsr = int(data['NOCSP'])
    cp2 = int(data['ECP2'])
    modeltpye=data['model']
    input = [[noa, nem, nsr, cp2]]

    if modeltpye == 'linear':
        predictions = modellinear.predict(input)
    elif modeltpye == 'SVR':
        predictions = svr.predict(input)
    elif modeltpye == 'DecisionTree':
        predictions = dt.predict(input)
    elif modeltpye == 'MLP':
        predictions = mlp.predict(input)
    elif modeltpye == 'SMO':
        s = scaler.transform(input)
        predictions = smo.predict(s)
    
    
    dict={'predictions':predictions[0],'model': modeltpye, 'status':200, 'statusText':'OK'}
    res=json.dumps(dict)
    return res

