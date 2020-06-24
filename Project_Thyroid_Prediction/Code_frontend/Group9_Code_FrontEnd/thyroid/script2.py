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
@app.route('/')

def home():
    return render_template('home.html')

#to tell flask what url shoud trigger the function index()
@app.route('/home')
def home2():
    return render_template('home.html')
@app.route('/intro' or '/form/intro' or '/team/intro' or '/result/intro'  or '/help/intro' or '/home/intro')
def intro():
    return render_template('intro.html')

@app.route('/form' or '/intro/form' or '/home/form' or '/result/intro' or '/help/form' or '/team/form')
def form():
    return render_template('form.html')

@app.route('/help' or '/home/help' or '/intro/help' or '/form/help' or '/team/help' or '/result/help')
def help():
    return render_template('help.html')

@app.route('/team' or '/home/team' or '/intro/team' or '/form/team' or '/result/team' or '/help/team')
def team():
    return render_template('team.html')

#@app.route('/intro/home')
#def home2():
#    return render_template('home.html')



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


