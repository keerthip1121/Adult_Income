# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 08:59:30 2021

@author: keerthi
"""


from flask import Flask,request,render_template
#import numpy as np
import pandas as pd
import joblib

# initialise flask
app = Flask(__name__,template_folder="template")

#load model
model = joblib.load("adult_income_catboost.pkl")

# launch home page
@app.route("/",methods = ["GET"])
def home():
    # load html page
    return render_template("css_template.html")
  
@app.route("/", methods = ["POST"]) 
def prediction():
    x_col = ["Age","Work_class","Education_num","Marital_Status","Occupation","Relationship","Race","Sex",
             "Capital_gain","Capital_loss","Hours_per_Week"]
    d = [[x for x in request.form.values()]]
    data = pd.DataFrame(d,columns=x_col)
    prediction = model.predict(data)[0]
    text = "Predicted Annual Income:"+str(prediction)
    return render_template("css_template.html", prediction = text) 
if __name__ == '__main__':
    app.run(debug=True)
