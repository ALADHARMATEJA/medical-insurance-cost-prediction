
from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
with open('stacking_regressor.pkl', 'rb') as f:
    stacking_regressor = pickle.load(f)

# Load the ColumnTransformer for encoding and scaling
with open('column_transformer.pkl', 'rb') as f:
    ct = pickle.load(f)

# Define the route for the default page
@app.route('/')
def home():
    return render_template('index.html',x=False)

# Define the route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']
    
    # Prepare the data for prediction
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    
    # Transform the input data using the column transformer
    input_data_transformed = ct.transform(input_data)
    
    # Make prediction
    prediction = stacking_regressor.predict(input_data_transformed)

    return render_template('index.html', prediction_text=f'Predicted Insurance Cost: {prediction[0]:.2f}',x=True)
    # return render_template('index.html', prediction_text=f'<div class="card"><div class="card-body">Predicted Insurance Cost: ${prediction[0]:.2f}</div></div>')


if __name__ == "__main__":
    app.run(debug=True)
