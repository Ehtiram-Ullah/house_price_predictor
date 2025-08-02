

from flask import Flask, request, render_template

import joblib

import numpy as np

import pandas as pd

app = Flask(__name__)

#Load the saved model
model = joblib.load("src\\model\\house_price_rf_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get Input values form 
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    parking = int(request.form['parking'])
    furnishingstatus = request.form['furnishingstatus']
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    prefarea = request.form['prefarea']


    # Create a feature array (must match training order)
    features = np.array([[area, bedrooms, bathrooms, stories, parking, furnishingstatus,
                          mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea]])
    
     # Predict log price and convert back to actual price
    pred_log = model.predict(features)
    pred_price = np.exp(pred_log)[0]

    return f"Predicted House Price: ${pred_price:,.2f}"

if __name__ == '__main__':
    app.run(debug=True)
