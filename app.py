"""
Created on Wed Sep 27 12:58:28 2023

@author: deepesh
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input values from the form
        age = float(request.form["Age"])
        gender = request.form["Gender"]
        location = request.form["Location"]
        subscription_length = float(request.form["Subscription_Length_Months"])
        monthly_bill = float(request.form["Monthly_Bill"])
        total_usage_gb = float(request.form["Total_Usage_GB"])

        # Create a DataFrame for the new customer data
        new_customer_data = pd.DataFrame({
            'Age': [age],
            'Subscription_Length_Months': [subscription_length],
            'Monthly_Bill': [monthly_bill],
            'Total_Usage_GB': [total_usage_gb],
            'Gender_Male': [1 if gender == 'Male' else 0],
            'Location_Los Angeles': [1 if location == 'Los Angeles' else 0],
            'Location_New York': [1 if location == 'New York' else 0],
            'Location_Miami': [1 if location == 'Miami' else 0],
            'Gender_Female': [1 if gender == 'Female' else 0],
            'Location_Other': [0]  # Assuming you have a default value
        })

        # Debugging: Print the new_customer_data
        print("Input Data:")
        print(new_customer_data)

        # Make predictions for the new customer
        churn_prediction = model.predict(new_customer_data)

        # Debugging: Print the prediction
        print("Churn Prediction:")
        print(churn_prediction)

        # Render the prediction on the HTML page
        return render_template("index.html", prediction_text="Churn Prediction: {}".format(churn_prediction[0]))
    
if __name__ == "__main__":
    flask_app.run(debug=True)

