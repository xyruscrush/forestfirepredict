from flask import Flask, request, jsonify,render_template
import pickle
import numpy as np
import pandas as pd
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
Standard_Scaler= pickle.load(open('models/scalar.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def index():
 return render_template('index.html')

@app.route('/predictdata', methods=['POST','GET'])
def predict_data():
    if request.method == 'POST':
       Temperature = float(request.form['Temperature'])
       RH=float( request.form['RH'])
       Ws= float(request.form['Ws'])
       Rain= float(request.form['Rain'])
       FFMC=float(request.form['FFMC'])
       DMC=float(request.form['DMC'])
       ISI=float(request.form['ISI'])
       Classes=float(request.form['Classes'])
       Region=float(request.form['Region'])
       new_data_scaled=Standard_Scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
       result=ridge_model.predict(new_data_scaled)
       print(result[0])
       return render_template('home.html',result=result[0])
    else:
       return render_template('home.html')

if (__name__ == '__main__'):
    app.run()