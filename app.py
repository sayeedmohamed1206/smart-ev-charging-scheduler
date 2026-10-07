
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
    model_path = Path(__file__).parent / "ev_project_bundle.pkl"
    return joblib.load(model_path)

try:
    bundle = load_bundle()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]
    medians = bundle["medians"]
    modes = bundle["modes"]

except Exception as e:
    st.error(
        "Could not load ev_project_bundle.pkl. "
        "Check that the file is in the same folder as app.py "
        "and that its saved keys match the code."
    )
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
# CALCULATE COST ACROSS TARIFF PERIODS
# --------------------------------------------------
def calculate_cost(start, duration, power):
    remaining_hours = float(duration)
    current_time = float(start)
    total_cost = 0.0

    while remaining_hours > 1e-9:
        hour = current_time % 24
        next_boundary = np.floor(current_time) + 1
        hours_to_boundary = next_boundary - current_time

        # Avoid zero-length steps near an hour boundary
        if hours_to_boundary <= 1e-9:
            current_time = next_boundary
            continue

        step = min(remaining_hours, hours_to_boundary)
        total_cost += power * step * tariff(int(np.floor(hour)))

        current_time += step
        remaining_hours -= step

    return round(total_cost, 2)


# --------------------------------------------------
# CORRECTED SMART SCHEDULER
# --------------------------------------------------
def find_best_slot(energy, power, current, departure):
    if power <= 0 or energy < 0:
        return None, None, None

    duration = energy / power

    # Charging must finish before departure.
    # A departure value of 24 means midnight.
    if current + duration > departure + 1e-9:
        return None, None, None

    # Baseline: begin charging immediately.
    immediate_cost = calculate_cost(current, duration, power)

    best_start = float(current)
    best_cost = immediate_cost

    # Test every feasible 15-minute start time.
    # Do not recommend a slot that finishes after departure.
    steps = int(np.floor((departure - current) * 4 + 1e-9))

    for step_number in range(1, steps + 1):
        start = current + step_number * 0.25

        if start + duration > departure + 1e-9:
            break

        cost = calculate_cost(start, duration, power)

        if cost < best_cost - 1e-9:
            best_cost = cost
            best_start = start

    return best_start, round(best_cost, 2), immediate_cost


# --------------------------------------------------
# TIME FORMATTING
# --------------------------------------------------
def format_time(hour_value):
    total_minutes = int(round(hour_value * 60)) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def format_duration(hours):
    total_minutes = int(round(hours * 60))
    return (
        f"{total_minutes // 60} hr "
        f"{total_minutes % 60} min"
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

if departure_hour <= current_hour:
    st.error("Departure must be after the current hour.")
    st.stop()

# --------------------------------------------------
# PREDICT AND OPTIMIZE
# --------------------------------------------------
if st.button("Predict and Optimize Charging", type="primary"):

    # Remaining battery capacity
    remaining_capacity = battery_capacity * (1 - soc / 100)

    # Build input using the training bundle's saved feature names
    input_data = {}

    for feature in feature_columns:
        if feature in numeric_features:
            input_data[feature] = medians[feature]
        elif feature in categorical_features:
            input_data[feature] = modes[feature]
        else:
            input_data[feature] = 0

    # Fill known battery-related features when present.
    # These checks avoid changing feature names expected by the model.
    for feature in feature_columns:
        name = feature.lower().replace("_", " ").strip()

        if "battery level" in name or "state of charge" in name:
            input_data[feature] = soc
        elif "battery capacity" in name:
            input_data[feature] = battery_capacity
        elif "energy consumed" in name or "energy required" in name:
            input_data[feature] = remaining_capacity
        elif "arrival time" in name:
            input_data[feature] = current_hour
        elif "departure time" in name:
            input_data[feature] = departure_hour
        elif "charging power" in name or "charger power" in name:
            input_data[feature] = charging_power

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error("Prediction failed. Check the model's expected input features.")
        st.exception(e)
        st.stop()

    # Prediction is treated as session energy in kWh.
    # Limit scheduling energy to the battery's remaining capacity.
    predicted_energy = max(0.0, prediction)
    energy_to_schedule = min(predicted_energy, remaining_capacity)

    duration = energy_to_schedule / charging_power

    if current_hour + duration > departure_hour + 1e-9:
        st.warning(
            "There may not be enough time to charge the required energy "
            "before departure at the selected power. Try an earlier "
            "departure setting only if appropriate, or a higher supported "
            "charging power."
        )

    best_start, smart_cost, immediate_cost = find_best_slot(
        energy_to_schedule,
        charging_power,
        current_hour,
        departure_hour
    )

    # --------------------------------------------------
    # PREDICTION RESULTS
    # --------------------------------------------------
    st.header("Prediction Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted session energy", f"{predicted_energy:.2f} kWh")
    c2.metric("Energy used for scheduling", f"{energy_to_schedule:.2f} kWh")
    c3.metric("Remaining battery capacity", f"{remaining_capacity:.2f} kWh")

    # --------------------------------------------------
    # SCHEDULE RESULTS
    # --------------------------------------------------
    st.header("Recommended Charging Schedule")

    if best_start is None:
        st.error(
            "No feasible charging schedule is available. "
            "The required energy cannot be charged before departure "
            "with the selected charging power."
        )
        st.stop()

    finish_time = best_start + duration

    c1, c2, c3 = st.columns(3)
    c1.metric("Charging duration", format_duration(duration))
    c2.metric("Recommended start", format_time(best_start))
    c3.metric("Estimated charging cost", f"₹{smart_cost:.2f}")

    st.write(f"**Scheduled finish:** {format_time(finish_time)}")

    # --------------------------------------------------
    # COST COMPARISON
    # --------------------------------------------------
    savings = max(0.0, immediate_cost - smart_cost)

    if immediate_cost > 0:
        savings_percent = savings / immediate_cost * 100
    else:
        savings_percent = 0.0

    if savings > 0:
        st.success(
            f"Recommended schedule saves approximately ₹{savings:.2f} "
            f"({savings_percent:.1f}%) compared with charging immediately."
        )
    else:
        st.info(
            "Charging immediately is already the cheapest feasible option "
            "under the configured tariff. Estimated savings: ₹0.00."
        )

    st.header("Cost Comparison")

    c1, c2, c3 = st.columns(3)
    c1.metric("Immediate charging", f"₹{immediate_cost:.2f}")
    c2.metric("Smart schedule", f"₹{smart_cost:.2f}")
    c3.metric("Estimated savings", f"₹{savings:.2f}")

    cost_df = pd.DataFrame({
        "Charging option": ["Immediate charging", "Smart schedule"],
        "Estimated cost (₹)": [immediate_cost, smart_cost]
    })

    st.bar_chart(
        cost_df.set_index("Charging option")
    )

    # --------------------------------------------------
    # TARIFF CHART
    # --------------------------------------------------
    st.header("Electricity Tariff by Hour")

    tariff_df = pd.DataFrame({
        "Hour": [f"{h:02d}:00" for h in range(24)],
        "Tariff (₹/kWh)": [tariff(h) for h in range(24)]
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

    st.caption(
        "Tariffs are illustrative, not live electricity prices. "
        "Actual costs depend on your provider's tariff, charging efficiency, "
        "power limits and vehicle behavior. The optimizer searches feasible "
        "start times in 15-minute increments."
    )
