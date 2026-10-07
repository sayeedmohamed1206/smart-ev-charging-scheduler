
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart EV Charging Scheduler",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Based Smart EV Charging Scheduler")
st.write(
    "Predict EV charging requirements and find a lower-cost "
    "charging schedule before departure."
)

# --------------------------------------------------
# LOAD TRAINED MODEL BUNDLE
# --------------------------------------------------
@st.cache_resource
def load_bundle():
    path = Path(__file__).parent / "ev_project_bundle.pkl"
    return joblib.load(path)


try:
    bundle = load_bundle()

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_columns = bundle["feature_columns"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]
    training_medians = bundle["training_medians"]
    training_modes = bundle["training_modes"]

except Exception as e:
    st.error("Unable to load the trained EV model bundle.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# ELECTRICITY TARIFF
# --------------------------------------------------
def tariff(hour):
    hour = int(hour) % 24

    if 0 <= hour < 6:
        return 5.0
    elif 6 <= hour < 17:
        return 7.0
    elif 17 <= hour < 22:
        return 10.0
    else:
        return 6.0


# --------------------------------------------------
# CALCULATE CHARGING COST
# --------------------------------------------------
def calculate_cost(start, duration, power):
    current = float(start)
    remaining = float(duration)
    total_cost = 0.0

    while remaining > 1e-9:
        hour = current % 24
        next_boundary = np.floor(current) + 1
        step_to_boundary = next_boundary - current

        if step_to_boundary <= 1e-9:
            current = next_boundary
            continue

        step = min(remaining, step_to_boundary)

        total_cost += (
            power * step * tariff(int(np.floor(hour)))
        )

        current += step
        remaining -= step

    return round(total_cost, 2)


# --------------------------------------------------
# FIND CHEAPEST FEASIBLE CHARGING SLOT
# --------------------------------------------------
def find_best_slot(energy, power, current, departure):
    if power <= 0 or energy < 0:
        return None, None, None

    duration = energy / power

    if current + duration > departure + 1e-9:
        return None, None, None

    immediate_cost = calculate_cost(current, duration, power)

    best_start = float(current)
    best_cost = immediate_cost

    # Check start times in 15-minute increments
    steps = int(np.floor((departure - current) * 4 + 1e-9))

    for i in range(1, steps + 1):
        start = current + i * 0.25

        if start + duration > departure + 1e-9:
            break

        cost = calculate_cost(start, duration, power)

        if cost < best_cost - 1e-9:
            best_cost = cost
            best_start = start

    return best_start, round(best_cost, 2), immediate_cost


# --------------------------------------------------
# FORMAT TIME
# --------------------------------------------------
def format_time(value):
    minutes = int(round(value * 60)) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_duration(hours):
    minutes = int(round(hours * 60))
    return f"{minutes // 60} hr {minutes % 60} min"


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("EV Charging Details")

battery_capacity = st.sidebar.number_input(
    "Battery capacity (kWh)",
    min_value=1.0,
    max_value=300.0,
    value=60.0,
    step=1.0
)

soc = st.sidebar.slider(
    "Current battery charge (%)",
    min_value=0,
    max_value=100,
    value=35
)

charging_power = st.sidebar.number_input(
    "Charging power (kW)",
    min_value=1.0,
    max_value=350.0,
    value=7.0,
    step=1.0
)

current_hour = st.sidebar.slider(
    "Current hour",
    min_value=0,
    max_value=23,
    value=16
)

departure_hour = st.sidebar.slider(
    "Departure hour",
    min_value=current_hour + 1,
    max_value=24,
    value=min(24, current_hour + 8)
)

st.sidebar.caption(
    "24-hour clock: departure hour 24 means midnight."
)

# --------------------------------------------------
# PREDICT AND OPTIMIZE
# --------------------------------------------------
if st.button("Predict and Optimize Charging", type="primary"):

    # Start with representative training values
    input_data = {}

    for feature in feature_columns:
        if feature in training_medians:
            input_data[feature] = training_medians[feature]
        elif feature in training_modes:
            input_data[feature] = training_modes[feature]
        else:
            st.error(f"Missing training default for feature: {feature}")
            st.stop()

    # Apply the values entered by the user
    input_data["Battery Capacity (kWh)"] = battery_capacity
    input_data["State of Charge (Start %)"] = soc
    input_data["Charging Rate (kW)"] = charging_power
    input_data["Start Hour"] = current_hour

    # Derive time categories from the selected start hour
    if 5 <= current_hour < 12:
        time_of_day = "Morning"
    elif 12 <= current_hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= current_hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    input_data["Time of Day"] = time_of_day

    # Use the selected hour's day-independent category defaults.
    # Other features retain training-set medians/modes.
    input_df = pd.DataFrame(
        [[input_data[col] for col in feature_columns]],
        columns=feature_columns
    )

    # Use the saved preprocessor before the trained model
    try:
        processed_input = preprocessor.transform(input_df)
        prediction = float(model.predict(processed_input)[0])

    except Exception as e:
        st.error(
            "Prediction failed. The saved model and preprocessor "
            "may expect a different input format."
        )
        st.exception(e)
        st.stop()

    # This app treats the model's prediction as session energy (kWh).
    # Verify this target matches the target used during model training.
    predicted_energy = max(0.0, prediction)

    remaining_capacity = battery_capacity * (1 - soc / 100)
    energy_to_schedule = min(predicted_energy, remaining_capacity)
    duration = energy_to_schedule / charging_power

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted session energy", f"{predicted_energy:.2f} kWh")
    col2.metric("Energy used for scheduling", f"{energy_to_schedule:.2f} kWh")
    col3.metric("Remaining battery capacity", f"{remaining_capacity:.2f} kWh")

    # --------------------------------------------------
    # SCHEDULE
    # --------------------------------------------------
    st.header("Recommended Charging Schedule")

    best_start, smart_cost, immediate_cost = find_best_slot(
        energy_to_schedule,
        charging_power,
        current_hour,
        departure_hour
    )

    if best_start is None:
        st.error(
            "The required charging session cannot finish before "
            "your selected departure. Try a higher supported charging "
            "power or a later departure."
        )
        st.stop()

    finish_time = best_start + duration

    col1, col2, col3 = st.columns(3)

    col1.metric("Charging duration", format_duration(duration))
    col2.metric("Recommended start", format_time(best_start))
    col3.metric("Estimated charging cost", f"₹{smart_cost:.2f}")

    st.write(f"**Scheduled finish:** {format_time(finish_time)}")

    # --------------------------------------------------
    # COST COMPARISON
    # --------------------------------------------------
    savings = max(0.0, immediate_cost - smart_cost)
    savings_percent = (
        savings / immediate_cost * 100
        if immediate_cost > 0
        else 0.0
    )

    if savings > 0:
        st.success(
            f"Recommended schedule saves approximately ₹{savings:.2f} "
            f"({savings_percent:.1f}%) compared with charging immediately."
        )
    else:
        st.info(
            "Charging immediately is the cheapest feasible option "
            "under the configured tariff. Estimated savings: ₹0.00."
        )

    st.header("Cost Comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("Immediate charging", f"₹{immediate_cost:.2f}")
    col2.metric("Smart schedule", f"₹{smart_cost:.2f}")
    col3.metric("Estimated savings", f"₹{savings:.2f}")

    cost_df = pd.DataFrame({
        "Charging option": ["Immediate charging", "Smart schedule"],
        "Estimated cost (₹)": [immediate_cost, smart_cost]
    })

    st.bar_chart(cost_df.set_index("Charging option"))

    # --------------------------------------------------
    # TARIFF CHART
    # --------------------------------------------------
    st.header("Electricity Tariff by Hour")

    tariff_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in range(24)],
        "Tariff (₹/kWh)": [tariff(h) for h in range(24)]
    })

    st.line_chart(tariff_df.set_index("Hour"))

    # --------------------------------------------------
    # TIMELINE
    # --------------------------------------------------
    st.header("Charging Schedule Timeline")

    timeline = pd.DataFrame({
        "Event": [
            "Current time",
            "Recommended charging start",
            "Charging finish",
            "Departure"
        ],
        "Time": [
            format_time(current_hour),
            format_time(best_start),
            format_time(finish_time),
            format_time(departure_hour)
        ]
    })

    st.table(timeline)

    st.caption(
        "Tariffs are illustrative, not live electricity prices. "
        "The scheduler checks feasible start times in 15-minute "
        "increments and assumes constant charging power. Actual costs "
        "depend on your provider's tariff, charging efficiency and "
        "vehicle behavior."
    )
