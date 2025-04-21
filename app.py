import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.models import load_model

# Load Model, scaler, encoders

model=load_model('Churn_Modelling.h5')
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
onehot_encoder_geo = pickle.load(open('OHE_geo.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))  

## TITLE OF THE APP
st.title("Customer Churn Prediction")

## USER INPUT
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make the prediction
prediction = model.predict(input_data_scaled)

# Display the prediction
st.write(f'Churn Probability: {prediction[0][0]:.2f}')

if prediction[0][0] > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")
    
