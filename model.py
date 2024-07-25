import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import joblib

file_path = 'new_updated_data.csv'
data = pd.read_csv(r"/Users/fahadzahid/Downloads/dps/dev_deployment/dps_AI_Engineer_Fahad/transformed.csv")
# Create DataFrame
df = pd.DataFrame(data)

# Initialize label encoders
label_encoder_monatszahl = LabelEncoder()
label_encoder_auspraegung = LabelEncoder()

# Fit and transform categorical features
df['MONATSZAHL'] = label_encoder_monatszahl.fit_transform(df['MONATSZAHL'])
df['AUSPRAEGUNG'] = label_encoder_auspraegung.fit_transform(df['AUSPRAEGUNG'])

# Define features and target
X = df[['MONATSZAHL', 'AUSPRAEGUNG', 'JAHR', 'MONAT']]
y = df['WERT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# print(f'r^2 Error: {r2}')

# Example prediction
example_data = pd.DataFrame({
    'MONATSZAHL': label_encoder_monatszahl.transform(['Alkoholunfälle']),
    'AUSPRAEGUNG': label_encoder_auspraegung.transform(['Verletzte und Getötete']),
    'JAHR': [2005],
    'MONAT': [6]
})
example_prediction = model.predict(example_data)
print(f'Predicted WERT for May 2020: {example_prediction[0]}')

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

# Save the label encoders
joblib.dump(label_encoder_monatszahl, 'label_encoder_monatszahl.pkl')
joblib.dump(label_encoder_auspraegung, 'label_encoder_auspraegung.pkl')
