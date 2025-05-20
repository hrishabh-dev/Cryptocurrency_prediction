from flask import Flask,jsonify,render_template,request
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)
# Load your scaler
with open('liquidity_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load your tuned model
with open('tuned_liquidity_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        # Prepare data for prediction
        volume = float(request.form['24h_volume'])
        mkt_cap = float(request.form['mkt_cap'])
        h1 = float(request.form['1h'])
        price = float(request.form['price'])

        # Create DataFrame with the same feature names as in your model
        input_dict = {
            '24h_volume': volume,
            'mkt_cap': mkt_cap,
            '1h': h1,
            'price': price
        }
        input_df = pd.DataFrame([input_dict])

        # Get the features the scaler expects
        scaler_features = scaler.feature_names_in_

        # Select only those columns, in the correct order
        input_df_scaled = input_df[scaler_features]

        # Scale the data
        scaled_input = scaler.transform(input_df_scaled)

        # Make prediction
        predicted_liquidity = loaded_model.predict(scaled_input)[0]
        liquidity_level = "High" if predicted_liquidity >= 0.5 else "Low"

        return render_template(
            "index.html",
            prediction_text=f"Predicted liquidity: {predicted_liquidity:.6f}",
            liquidity_level=f"Liquidity Level: {liquidity_level}"
        )
    else:
        return render_template("index.html")






if __name__=="__main__":
    app.run(debug=True)