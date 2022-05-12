from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/home.html', methods = ['POST', 'GET'])
def a():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        pred = "Mammal"
    elif prediction == 2:
        pred = "Bird"
    elif prediction == 3:
        pred = "Reptile"
        
    elif prediction == 4:
        pred = "Fish"
    elif prediction == 5:
        pred = "Amphibian"
        
    elif prediction == 6:
        pred = "Bug"
        
    elif prediction == 7:
        pred = "Invertebrate"
        
    else: 
         pred = 'Unknown class'
    output = pred
    return render_template("home.html",  prediction_text=' it is a {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)