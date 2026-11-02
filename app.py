import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Must be the first Streamlit command in the script
st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒")

# ------------------------------------------------------------------
# Important: these helper functions must match the notebook exactly
# (same names, same logic), because the saved pipeline looks them up
# when it is loaded.
# ------------------------------------------------------------------
code_cols = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def add_features(d):
    d = d.copy()
    d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
    d["TotalDuration"] = (d["Administrative_Duration"] + d["Informational_Duration"]
                          + d["ProductRelated_Duration"])
    d["AvgTimePerPage"] = (d["TotalDuration"] / d["TotalPages"].replace(0, np.nan)).fillna(0)
    return d


def codes_to_str(d):
    d = d.copy()
    d[code_cols] = d[code_cols].astype(str)
    return d


@st.cache_resource
def load_model():
    bundle = joblib.load("model.joblib")
    return bundle["pipeline"], bundle["threshold"]


pipeline, THRESHOLD = load_model()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🛒 Will This Visitor Buy?")
st.write(
    "Enter a visitor's browsing-session details. The app estimates the "
    "probability of a purchase and suggests an action."
)

st.subheader("Visitor activity")
c1, c2 = st.columns(2)
with c1:
    administrative = st.number_input("Administrative pages (account/login pages)", 0, 100, 0)
    informational = st.number_input("Informational pages (about/contact)", 0, 100, 0)
    product_related = st.number_input("Product pages visited", 0, 1000, 10)
with c2:
    administrative_dur = st.number_input("Administrative duration (seconds)", 0.0, 10000.0, 0.0)
    informational_dur = st.number_input("Informational duration (seconds)", 0.0, 10000.0, 0.0)
    product_related_dur = st.number_input("Product pages duration (seconds)", 0.0, 100000.0, 300.0)

st.subheader("Website metrics")
c3, c4 = st.columns(2)
with c3:
    bounce = st.slider("Bounce rate", 0.0, 0.2, 0.02, 0.001,
                       help="Share of visitors who land on one page and leave.")
    exit_rate = st.slider("Exit rate", 0.0, 0.2, 0.04, 0.001,
                          help="Share of exits from the pages the visitor viewed.")
with c4:
    page_values = st.number_input(
        "Page value", 0.0, 400.0, 0.0,
        help="Average value of the pages visited (comes from Google Analytics). "
             "It is the most important factor in the model. "
             "If you don't have it, leave it at 0.")
    special_day = st.select_slider("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)

st.subheader("Visitor details")
c5, c6 = st.columns(2)
with c5:
    month = st.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 8)
    visitor_type = st.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("Weekend")
with c6:
    os_ = st.number_input("Operating system (code)", 1, 8, 2)
    browser = st.number_input("Browser (code)", 1, 13, 2)
    region = st.number_input("Region (code)", 1, 9, 1)
    traffic = st.number_input("Traffic type (code)", 1, 20, 2)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_dur,
        "Informational": informational,
        "Informational_Duration": informational_dur,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": int(os_),
        "Browser": int(browser),
        "Region": int(region),
        "TrafficType": int(traffic),
        "VisitorType": visitor_type,
        "Weekend": int(weekend),
    }])

    try:
        proba = float(pipeline.predict_proba(row)[0, 1])
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")
        st.stop()

    st.subheader("Result")
    st.metric("Purchase probability", f"{proba:.1%}")
    st.progress(min(max(proba, 0.0), 1.0))

    # Example action policy - adjust the cut-offs based on your own analysis
    if proba >= THRESHOLD:
        st.success("✅ This visitor is likely to buy. **No discount needed.**")
    elif proba >= THRESHOLD * 0.4:
        st.warning("🎟️ This visitor is undecided. **Show a discount coupon.**")
    else:
        st.info("💬 Purchase probability is low. **Offer the chat assistant** to help them find what they need.")

    st.caption(f"Model decision threshold = {THRESHOLD:.2f}")

st.divider()
st.caption(
    "For estimation only. Results are based on the UCI Online Shoppers "
    "Purchasing Intention dataset."
)
