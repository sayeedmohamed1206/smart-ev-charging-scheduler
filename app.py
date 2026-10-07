import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Smart EV Charging Scheduler",
    page_icon="🔋",
    layout="wide"
)

bundle = joblib.load("ev_project_bundle.pkl")

model = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
medians = bundle["training_medians"]
modes = bundle["training_modes"]


def get_tariff(hour):
    if 0 <= hour < 6:
        return 5.0
    elif 6 <= hour < 17:
        return 7.0
    elif 17 <= hour < 22:
        return 10.0
    return 6.0


def calculate_cost(start, duration, power):
    total = 0.0
    t = start
    end = start + duration

    while t < end - 1e-9:
        next_t = min(int(t) + 1, end)
        energy = power * (next_t - t)
        total += energy * get_tariff(int(t) % 24)
        t = next_t

    return total


def find_best_slot(energy, power, current, departure):
    duration = energy / power
    slots = []
    start = float(current)

    while start + duration <= departure + 1e-9:
        cost = calculate_cost(start, duration, power)
        slots.append((cost, start))
        start += 0.25

    if not slots:
        return None, duration

    cost, start = min(slots)
    return {
        "start": start,
        "end": start + duration,
        "cost": cost
    }, duration


st.title("🔋 Smart EV Charging Scheduler")
st.caption("AI-based energy prediction and tariff-aware scheduling")

st.info(
    "Prototype note: electricity prices are illustrative. "
    "Recommendations depend on the trained dataset model and "
    "simplified constant-power charging assumptions."
)

with st.sidebar:
    st.header("EV Charging Details")
    battery = st.number_input(
        "Battery capacity (kWh)", 5.0, 200.0, 60.0
    )
    soc = st.slider("Current battery charge (%)", 0, 99, 35)
    power = st.number_input(
        "Charging power (kW)", 1.0, 50.0, 7.0
    )
    current = st.slider("Current hour", 0, 22, 16)
    departure = st.slider(
        "Departure hour", current + 1, 24, min(23, current + 7)
    )

    st.caption("Hours use a 24-hour clock. Departure hour 24 means midnight.")

if st.button("Predict and Optimize Charging", type="primary"):
    row = {}

    for col in feature_columns:
        if col in numeric_features:
            row[col] = medians.get(col, 0.0)
        else:
            row[col] = modes.get(col, "Unknown")

    supplied = {
        "Battery Capacity (kWh)": battery,
        "State of Charge (Start %)": soc,
        "Charging Rate (kW)": power,
        "Start Hour": current,
        "Start Month": 1,
        "Start Day of Week": 0
    }

    for col, value in supplied.items():
        if col in row:
            row[col] = value

    input_df = pd.DataFrame([row], columns=feature_columns)
    transformed = preprocessor.transform(input_df)
    predicted = max(0.0, float(model.predict(transformed)[0]))

    # The remaining battery capacity is a physical upper bound.
    remaining = battery * (100 - soc) / 100
    energy = min(predicted, remaining)

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted session energy", f"{predicted:.2f} kWh")
    c2.metric("Energy used for scheduling", f"{energy:.2f} kWh")
    c3.metric("Available battery capacity", f"{remaining:.2f} kWh")

    if energy <= 0:
        st.warning("No additional charging energy is required.")
    else:
        best, duration = find_best_slot(
            energy, power, current, departure
        )

        if best is None:
            st.error(
                "The required energy cannot be delivered before "
                "the selected departure time. Choose an earlier "
                "current hour, later departure, or lower energy requirement."
            )
        else:
            immediate = calculate_cost(current, duration, power)
            savings = immediate - best["cost"]
            percent = savings / immediate * 100 if immediate else 0

            st.subheader("Recommended Charging Schedule")
            a, b, c = st.columns(3)
            a.metric("Charging duration", f"{duration:.2f} hours")
            b.metric("Recommended start", f"{best['start']:.2f}:00")
            c.metric("Estimated cost", f"₹{best['cost']:.2f}")

            st.subheader("Cost Comparison")
            d, e, f = st.columns(3)
            d.metric("Immediate charging", f"₹{immediate:.2f}")
            e.metric("Smart schedule", f"₹{best['cost']:.2f}")
            f.metric("Estimated savings", f"₹{savings:.2f}",
                     delta=f"{percent:.1f}%")

            chart = pd.DataFrame({
                "Method": ["Immediate charging", "Smart schedule"],
                "Estimated Cost (₹)": [immediate, best["cost"]]
            }).set_index("Method")
            st.bar_chart(chart)

            st.caption(
                "Predictions are model estimates, and costs are "
                "calculated from illustrative tariffs. Actual savings "
                "depend on real tariffs, charging efficiency and vehicle limits."
            )

st.divider()
st.caption("MSc AI/ML Project | Smart EV Charging Scheduler")
