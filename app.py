import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CarValue", page_icon="🚗")

@st.cache_resource
def load():
    return joblib.load("carvalue_model.joblib"), joblib.load("meta.joblib")

model, meta = load()

st.title("🚗 CarValue: Used Car Price Estimator")
st.write("Enter the car's details to get an estimated resale price.")

car_model = st.selectbox("Car model", meta["models"])
brand = meta["model_to_brand"][car_model]
st.caption(f"Brand: {brand.title()}")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Manufacturing year", meta["year_min"], meta["current_year"],
                           value=min(2018, meta["current_year"]))
    km = st.number_input("Kilometres driven", 1, 1_000_000, 50_000, step=1000)
    owner = st.selectbox("Ownership", ["first owner", "second owner",
                                       "third owner", "fourth & above owner"])
with col2:
    fuel = st.selectbox("Fuel", meta["fuels"])
    transmission = st.selectbox("Transmission", meta["transmissions"])
    seller = st.selectbox("Seller type", meta["seller_types"])

if st.button("Estimate price"):
    if year > meta["year_max"]:
        st.warning(f"The training data only includes cars up to {meta['year_max']}. "
                   "Estimates for newer cars are unreliable.")
    if year < 2000:
        st.warning("Very old cars are rare in the training data, so this estimate is less reliable.")
    if km > meta["km_max"]:
        st.warning("Kilometres are beyond the data the model was trained on. Treat this estimate with extra caution.")

    row = pd.DataFrame([{
        "car_age": meta["current_year"] - year,
        "km_driven": km,
        "owner_num": meta["owner_map"][owner],
        "brand": brand, "model": car_model,
        "fuel": fuel, "transmission": transmission, "seller_type": seller,
    }])
    p = model.predict(row)[0]
    lo, hi = p * np.exp(meta["q10"]), p * np.exp(meta["q90"])

    st.subheader(f"Estimated price: ₹{p:,.0f}")
    st.write(f"Likely range: **₹{lo:,.0f} to ₹{hi:,.0f}**")

st.info("This is an estimate, not a guaranteed selling price. It is based on historical "
        "listing prices and does not account for condition, accident history, colour, "
        "or local demand.")