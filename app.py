import pickle
from flask import Flask,render_template,request,app,jsonify,url_for
import pandas as pd
import numpy as np

app=Flask(__name__) #starting point of application from where it will run
#load the model
model=pickle.load(open(r'E:\Python Projects\BostonHousing\models\regressionModel.pkl','rb')) 

@app.route('/') #home page
def home():
    return render_template('home.html')
#  i.e. the moment flask app gets intiated, it will re-direct to the home.html (generally, home.html is a welcome page)

@app.route('/predict_api',methods=['POST']) #we give some input and its captured by post and post send it to predict_api

def predict_api():
    data=request.json['data']
    # whenever the input is given in json format which is captured inside the 'data' key
    #from here when the post request is hit, its captured using request.json and sotred in data
    return jsonify(model.predict(np.array(list(data.values())).reshape(1,-1))[0])

if __name__=="__main__":
    app.run(debug=True)
    
    