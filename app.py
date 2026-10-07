
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
# LOAD MODEL BUNDLE
# --------------------------------------------------
@st.cache_resource
def load_bundle():
    path = Path(__file__).parent / "ev_project_bundle.pkl"
    return joblib.load(path)


try:
    bundle = load_bundle()

    # Required saved model and feature information
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Optional keys: do not crash if they are absent
    numeric_features = bundle.get("numeric_features", [])
    categorical_features = bundle.get("categorical_features", [])
    medians = bundle.get("medians", {})
    modes = bundle.get("modes", {})

except Exception as e:
    st.error("Could not load ev_project_bundle.pkl.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# TARIFF RATES
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
# CHARGING COST CALCULATION
# --------------------------------------------------
def calculate_cost(start, duration, power):
    remaining = float(duration)
    current = float(start)
    total_cost = 0.0

    while remaining > 1e-9:
        hour = current % 24
        next_boundary = np.floor(current) + 1
        time_to_boundary = next_boundary - current

        if time_to_boundary <= 1e-9:
            current = next_boundary
            continue

        step = min(remaining, time_to_boundary)

        total_cost += (
            power
            * step
            * tariff(int(np.floor(hour)))
        )

        current += step
        remaining -= step

    return round(total_cost, 2)


# --------------------------------------------------
# OPTIMAL CHARGING SCHEDULER
# --------------------------------------------------
def find_best_slot(energy, power, current, departure):
    if power <= 0 or energy < 0:
        return None, None, None

    duration = energy / power

    # Ensure charging can finish before departure
    if current + duration > departure + 1e-9:
        return None, None, None

    immediate_cost = calculate_cost(
        current, duration, power
    )

    best_start = float(current)
    best_cost = immediate_cost

    # Check every 15-minute feasible start time
    number_of_steps = int(
        np.floor((departure - current) * 4 + 1e-9)
    )

    for i in range(1, number_of_steps + 1):
        start = current + i * 0.25

        if start + duration > departure + 1e-9:
            break

        cost = calculate_cost(start, duration, power)

        # Keep the cheapest feasible schedule
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_start = start

    return best_start, round(best_cost, 2), immediate_cost


# --------------------------------------------------
# TIME DISPLAY
# --------------------------------------------------
def format_time(hour_value):
    minutes = int(round(hour_value * 60)) % 1440
    hour = minutes // 60
    minute = minutes % 60

    return f"{hour:02d}:{minute:02d}"


def format_duration(hours):
    minutes = int(round(hours * 60))

    return (
        f"{minutes // 60} hr "
        f"{minutes % 60} min"
    )


# --------------------------------------------------
# USER INPUTS
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
    "Use a 24-hour clock. Departure hour 24 means midnight."
)


# --------------------------------------------------
# PREDICTION AND OPTIMIZATION
# --------------------------------------------------
if st.button("Predict and Optimize Charging"):

    remaining_capacity = (
        battery_capacity * (1 - soc / 100)
    )

    # Build model input using saved preprocessing values
    input_data = {}

    for feature in feature_columns:
        if feature in medians:
            input_data[feature] = medians[feature]
        elif feature in modes:
            input_data[feature] = modes[feature]
        else:
            input_data[feature] = 0

    # Set known input features where the names match
    for feature in feature_columns:
        name = feature.lower().replace("_", " ").strip()

        if (
            "battery level" in name
            or "state of charge" in name
            or name == "soc"
        ):
            input_data[feature] = soc

        elif "battery capacity" in name:
            input_data[feature] = battery_capacity

        elif (
            "energy consumed" in name
            or "energy required" in name
        ):
            input_data[feature] = remaining_capacity

        elif "arrival time" in name:
            input_data[feature] = current_hour

        elif "departure time" in name:
            input_data[feature] = departure_hour

        elif (
            "charging power" in name
            or "charger power" in name
        ):
            input_data[feature] = charging_power

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # Make prediction
    try:
        prediction = float(model.predict(input_df)[0])

    except Exception as e:
        st.error(
            "Prediction failed. The saved model's expected input "
            "features may differ from the app."
        )
        st.exception(e)
        st.stop()

    predicted_energy = max(0.0, prediction)

    # Limit energy to the battery's remaining capacity
    energy_to_schedule = min(
        predicted_energy,
        remaining_capacity
    )

    duration = energy_to_schedule / charging_power

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted session energy",
        f"{predicted_energy:.2f} kWh"
    )

    col2.metric(
        "Energy used for scheduling",
        f"{energy_to_schedule:.2f} kWh"
    )

    col3.metric(
        "Remaining battery capacity",
        f"{remaining_capacity:.2f} kWh"
    )

    # --------------------------------------------------
    # FIND BEST SCHEDULE
    # --------------------------------------------------
    best_start, smart_cost, immediate_cost = find_best_slot(
        energy_to_schedule,
        charging_power,
        current_hour,
        departure_hour
    )

    st.header("Recommended Charging Schedule")

    if best_start is None:
        st.error(
            "No feasible schedule is available. The charging "
            "session cannot finish before your selected departure. "
            "Try a higher supported charging power or a later "
            "departure time."
        )
        st.stop()

    finish_time = best_start + duration

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Charging duration",
        format_duration(duration)
    )

    col2.metric(
        "Recommended start",
        format_time(best_start)
    )

    col3.metric(
        "Estimated charging cost",
        f"₹{smart_cost:.2f}"
    )

    st.write(
        f"**Scheduled finish:** {format_time(finish_time)}"
    )

    # --------------------------------------------------
    # COST COMPARISON
    # --------------------------------------------------
    savings = max(
        0.0,
        immediate_cost - smart_cost
    )

    savings_percent = (
        savings / immediate_cost * 100
        if immediate_cost > 0
        else 0.0
    )

    if savings > 0:
        st.success(
            f"Recommended schedule saves approximately "
            f"₹{savings:.2f} ({savings_percent:.1f}%) "
            f"compared with charging immediately."
        )
    else:
        st.info(
            "Charging immediately is already the cheapest "
            "feasible option under the configured tariff. "
            "Estimated savings: ₹0.00."
        )

    st.header("Cost Comparison")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Immediate charging",
        f"₹{immediate_cost:.2f}"
    )

    col2.metric(
        "Smart schedule",
        f"₹{smart_cost:.2f}"
    )

    col3.metric(
        "Estimated savings",
        f"₹{savings:.2f}"
    )

    cost_df = pd.DataFrame({
        "Charging option": [
            "Immediate charging",
            "Smart schedule"
        ],
        "Estimated cost (₹)": [
            immediate_cost,
            smart_cost
        ]
    })

    st.bar_chart(
        cost_df.set_index("Charging option")
    )

    # --------------------------------------------------
    # TARIFF CHART
    # --------------------------------------------------
    st.header("Electricity Tariff by Hour")

    tariff_df = pd.DataFrame({
        "Hour": [
            f"{hour:02d}:00"
            for hour in range(24)
        ],
        "Tariff (₹/kWh)": [
            tariff(hour)
            for hour in range(24)
        ]
    })

    st.line_chart(
        tariff_df.set_index("Hour")
    )

    # --------------------------------------------------
    # CHARGING TIMELINE
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

    # --------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------
    st.caption(
        "Tariffs are illustrative, not live electricity prices. "
        "The scheduler assumes constant charging power and "
        "checks feasible start times in 15-minute increments. "
        "Actual costs depend on the electricity tariff, charging "
        "efficiency, vehicle behavior and charging limits."
    )
