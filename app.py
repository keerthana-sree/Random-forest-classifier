import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="ML Prediction App", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival")


@st.cache_resource
def load_model():
    with open("titanic_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0.0, 500.0, 50.0)


sex = 1 if sex == "Male" else 0


if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Passenger is likely to SURVIVE")
    else:
        st.error("❌ Passenger is NOT likely to survive")
