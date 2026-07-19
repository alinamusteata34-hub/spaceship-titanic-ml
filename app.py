import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('spaceship_model.pkl')

st.title("🚀 Spaceship Titanic — Transported Predictor")
st.write("Enter passenger details to predict whether they were transported to an alternate dimension.")

# Input form
home_planet = st.selectbox("Home Planet", ["Earth", "Europa", "Mars"])
cryo_sleep = st.selectbox("CryoSleep", ["False", "True"])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"])
age = st.slider("Age", 0, 100, 25)
vip = st.selectbox("VIP", ["False", "True"])

st.subheader("Spending")
room_service = st.number_input("Room Service ($)", min_value=0, value=0)
food_court = st.number_input("Food Court ($)", min_value=0, value=0)
shopping_mall = st.number_input("Shopping Mall ($)", min_value=0, value=0)
spa = st.number_input("Spa ($)", min_value=0, value=0)
vr_deck = st.number_input("VR Deck ($)", min_value=0, value=0)

st.subheader("Cabin")
deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
cabin_num = st.number_input("Cabin Number", min_value=0, value=0)
side = st.selectbox("Side", ["P", "S"])

# Encode categorical inputs to match training (same LabelEncoder order used before)
home_planet_map = {"Earth": 0, "Europa": 1, "Mars": 2}
cryo_map = {"False": 0, "True": 1}
destination_map = {"55 Cancri e": 0, "PSO J318.5-22": 1, "TRAPPIST-1e": 2}
vip_map = {"False": 0, "True": 1}
deck_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
side_map = {"P": 0, "S": 1}

# Build input row in the exact same feature order as training
input_data = pd.DataFrame([{
    "HomePlanet": home_planet_map[home_planet],
    "CryoSleep": cryo_map[cryo_sleep],
    "Destination": destination_map[destination],
    "Age": age,
    "VIP": vip_map[vip],
    "RoomService": room_service,
    "FoodCourt": food_court,
    "ShoppingMall": shopping_mall,
    "Spa": spa,
    "VRDeck": vr_deck,
    "Deck": deck_map[deck],
    "CabinNum": cabin_num,
    "Side": side_map[side]
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction:
        st.success(f"✅ Predicted: Transported! (Confidence: {probability:.1%})")
    else:
        st.error(f"❌ Predicted: Not Transported (Confidence: {(1-probability):.1%})")
