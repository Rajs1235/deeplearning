import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('churn_model.h5')

# Load preprocessing objects
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# Streamlit UI
st.title("Customer Churn Prediction")

st.write("Enter customer details:")

geography = st.selectbox(
    "Geography",
    onehot_encoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder.classes_
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=3
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=0.0
)

num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# ----------------------------------
# Prepare Input Data
# ----------------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Gender
input_data['Gender'] = label_encoder.transform(
    input_data['Gender']
)

# Encode Geography
geo_encoded = onehot_encoder.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder.get_feature_names_out(['Geography'])
)

# Combine
input_df = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Match exact training column order
input_df = input_df.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

prediction_proba = float(prediction[0][0])

st.subheader("Prediction")

if prediction_proba > 0.5:
    st.error(
        f"Customer is likely to churn.\n\nProbability: {prediction_proba:.2%}"
    )
else:
    st.success(
        f"Customer is unlikely to churn.\n\nProbability: {prediction_proba:.2%}"
    )

# Debug (remove later)
with st.expander("Debug Information"):
    st.write("Model Input Columns:")
    st.write(input_df.columns.tolist())

    st.write("Scaler Expected Columns:")
    st.write(list(scaler.feature_names_in_))

    st.write("Input DataFrame:")
    st.dataframe(input_df)