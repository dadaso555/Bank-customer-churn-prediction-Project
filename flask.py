import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Example data and model training (replace with your actual data and model)
data = {
    'CreditScore': [600, 700],
    'Age': [40, 50],
    'Tenure': [3, 5],
    'Balance': [60000, 80000],
    'NumOfProducts': [2, 1],
    'HasCrCard': [1, 0],
    'IsActiveMember': [1, 1],
    'EstimatedSalary': [50000, 60000],
    'Geography_Germany': [1, 0],
    'Geography_Spain': [0, 1],
    'Gender_Male': [1, 0],
    'Exited': [0, 1]  # Target variable
}

df = pd.DataFrame(data)

# Features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train a model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model to a joblib file
joblib.dump(model, 'Customer_Churn_Prediction_model.joblib')

import streamlit as st
import pandas as pd
import joblib

st.title("Bank Churn Prediction")

# Load the model using joblib
loaded_model = joblib.load('Customer_Churn_Prediction_model.joblib')

# User input
st.sidebar.title('Customer Data')
CreditScore = st.sidebar.number_input('CreditScore', min_value=350, max_value=850, step=1)
Age = st.sidebar.number_input('Age', min_value=18, max_value=92, step=1)
Tenure = st.sidebar.number_input('Tenure', min_value=0, max_value=10, step=1)
Balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=250898.09, step=1.0)
NumOfProducts = st.sidebar.number_input('NumOfProducts', min_value=0, max_value=4, step=1)
HasCrCard = st.sidebar.radio('HasCrCard', ('Yes', 'No'))
IsActiveMember = st.sidebar.radio('IsActiveMember', ('Yes', 'No'))
EstimatedSalary = st.sidebar.number_input('EstimatedSalary', min_value=11.58, max_value=199992.48, step=1.0)
Geography_Germany = st.sidebar.radio('Geography_Germany', ('Yes', 'No'))
Geography_Spain = st.sidebar.radio('Geography_Spain', ('Yes', 'No'))
Gender_Male = st.sidebar.radio('Gender_Male', ('Yes', 'No'))

# Convert user input to DataFrame
user_data = pd.DataFrame({
    'CreditScore': [CreditScore],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [1 if HasCrCard == 'Yes' else 0],
    'IsActiveMember': [1 if IsActiveMember == 'Yes' else 0],
    'EstimatedSalary': [EstimatedSalary],
    'Geography_Germany': [1 if Geography_Germany == 'Yes' else 0],
    'Geography_Spain': [1 if Geography_Spain == 'Yes' else 0],
    'Gender_Male': [1 if Gender_Male == 'Yes' else 0]
})

# Convert numerical values to corresponding labels for display
user_data_display = user_data.copy()
user_data_display['HasCrCard'] = 'Yes' if user_data_display['HasCrCard'].iloc[0] == 1 else 'No'
user_data_display['IsActiveMember'] = 'Yes' if user_data_display['IsActiveMember'].iloc[0] == 1 else 'No'
user_data_display['Geography_Germany'] = 'Yes' if user_data_display['Geography_Germany'].iloc[0] == 1 else 'No'
user_data_display['Geography_Spain'] = 'Yes' if user_data_display['Geography_Spain'].iloc[0] == 1 else 'No'
user_data_display['Gender_Male'] = 'Yes' if user_data_display['Gender_Male'].iloc[0] == 1 else 'No'

# Display user input and prediction result
st.subheader('User Input Data')
st.write(user_data_display)

st.subheader('Prediction Result')

# Prediction
if st.sidebar.button('Predict'):
    try:
        prediction = loaded_model.predict(user_data)
        if prediction[0] == 1:
            result = 'Customer will Churn'
            st.success(result)
        else:
            result = 'Customer will not Churn'
            st.error(result)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
