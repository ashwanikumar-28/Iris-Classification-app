import streamlit as st
import pickle
import numpy as np

with open("iris_model3.pkl", "rb") as f:
    model = pickle.load(f)

with open("iris_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🌸 Iris Flower Classifier")

st.write("Enter the flower measurements:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)


if st.button("Predict"):
    
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    st.success(f"The predicted Iris species is: **{prediction}**")