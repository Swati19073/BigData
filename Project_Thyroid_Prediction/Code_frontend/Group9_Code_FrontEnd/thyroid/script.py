#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/form')
def index():
    return flask.render_template('form.html')


# In[ ]:

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]



@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        new_list=[]
        for i in to_predict_list:
            new_list.append(float(i))
        #to_predict_list = list(map(float, to_predict_list))
        #result = ValuePredictor(to_predict_list)
        result= ValuePredictor(new_list)
        if int(result)==1:
            prediction='You do not have Thyroid :D '
        elif int(result)==2:
            prediction='You are predicted with Hyperthyroidism :('
        else:
            prediction='You are predicted with Hypothyroidism :('
        
        
        return render_template("result.html",prediction=prediction)


