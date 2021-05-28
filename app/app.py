from flask import Flask, render_template, request
import numpy as np
from joblib import load

model = load('app/model.joblib')

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/predict", methods = ['POST','GET'])
def predict():
    features = []
    features.append(float(request.form['Cement']))
    features.append(np.log1p(float(request.form['BlastFurnaceSlag'])))
    features.append(np.log1p(float(request.form['FlyAsh'])))
    features.append(np.log1p(float(request.form['Water'])))
    features.append(np.log1p(float(request.form['Superplastisizer'])))
    features.append(float(request.form['CoarseAggregate']))
    features.append(float(request.form['FineAggregate']))
    features.append(np.log1p(float(request.form['Age'])))

    final = np.array(features)
    prediction = model.predict(final)
    return render_template('index.html', pred = 'The predicted Concrete Compressive strength is {:0.2f} (+/-2.75) MPA '.format(prediction))




