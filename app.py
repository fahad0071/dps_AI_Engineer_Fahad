from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the model and encoders
model = joblib.load('random_forest_model.pkl')
label_encoder_monatszahl = joblib.load('label_encoder_monatszahl.pkl')
label_encoder_auspraegung = joblib.load('label_encoder_auspraegung.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get data from POST request

        # Map the incoming data to the expected feature names
        df = pd.DataFrame({
            'MONATSZAHL': [data['Category']],
            'AUSPRAEGUNG': [data['Type']],
            'JAHR': [int(data['Year'])],
            'MONAT': [int(data['Month'])]
        })

        # Encode categorical features
        df['MONATSZAHL'] = label_encoder_monatszahl.transform(df['MONATSZAHL'])
        df['AUSPRAEGUNG'] = label_encoder_auspraegung.transform(df['AUSPRAEGUNG'])
        
        # Define features
        X = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT']]
        
        # Predict
        predictions = model.predict(X)
        
        # Ensure there's only one prediction and format the response
        if len(predictions) == 1:
            response = {
                "prediction": float(predictions[0])  # Convert to float for JSON serialization
            }
        else:
            response = {
                "error": "Expected a single prediction but received multiple values."
            }
    
        return jsonify({
                    'status': 200,
                    'success': True,
                    'message': 'Model sucessfully executed',
                    'data': response
                }), 200
    except Exception as e:
        return jsonify({
            'status': 500,
            'success': False,
            'message': 'An error occurred',
            'example': "'Category': 'Alkoholunfälle','Type': 'insgesamt','Year': '2021','Month': '01'",
            'details':f"(Missing parameter: {str(e)})"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
