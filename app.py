
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart EV Charging Scheduler",
    page_icon="🔋",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD SAVED MACHINE LEARNING MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "ev_project_bundle.pkl"

    if not model_path.exists():
        st.error("Model file not found: ev_project_bundle.pkl")
        st.stop()

    return joblib.load(model_path)


bundle = load_model()

model = bundle["model"]
preprocessor = bundle["preprocessor"]
feature_columns = bundle["feature_columns"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]
medians = bundle["training_medians"]
modes = bundle["training_modes"]

# --------------------------------------------------
# 3. ILLUSTRATIVE ELECTRICITY TARIFF
# --------------------------------------------------
def get_tariff(hour):
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
# 4. CALCULATE COST ACROSS HOURLY TARIFF PERIODS
# --------------------------------------------------
def calculate_cost(start, duration, power):
    total_cost = 0.0
    end = start + duration
    t = start

    while t < end - 1e-9:
        hour = int(t) % 24
        next_hour = np.floor(t) + 1
        next_t = min(next_hour, end)

        energy_used = power * (next_t - t)
        total_cost += energy_used * get_tariff(hour)

        t = next_t

    return total_cost


# --------------------------------------------------
# 5. FIND THE LOWEST-COST FEASIBLE CHARGING SLOT
# --------------------------------------------------
def find_best_slot(energy, power, current, departure):
    if energy <= 0 or power <= 0 or departure <= current:
        return None, 0.0

    duration = energy / power

    if duration > departure - current + 1e-9:
        return None, duration

    slots = []
    start = float(current)

    # Test possible start times every 15 minutes.
    while start + duration <= departure + 1e-9:
        cost = calculate_cost(start, duration, power)

        slots.append({
            "start": start,
            "end": start + duration,
            "cost": cost
        })

        start += 0.25

    if not slots:
        return None, duration

    # Select the lowest-cost feasible option.
    best = min(slots, key=lambda item: item["cost"])

    return best, duration


# --------------------------------------------------
# 6. FORMAT TIME CORRECTLY
# --------------------------------------------------
def format_time(hour):
    total_minutes = round(hour * 60)
    hours = (total_minutes // 60) % 24
    minutes = total_minutes % 60

    return f"{hours:02d}:{minutes:02d}"


def format_duration(hours):
    total_minutes = round(hours * 60)
    whole_hours = total_minutes // 60
    minutes = total_minutes % 60

    if whole_hours:
        return f"{whole_hours} hr {minutes} min"

    return f"{minutes} min"


# --------------------------------------------------
# 7. APPLICATION HEADER
# --------------------------------------------------
st.title("🔋 Smart EV Charging Scheduler")
st.caption(
    "AI-based energy prediction and tariff-aware charging optimization"
)

st.info(
    "Project prototype: electricity tariffs are illustrative. "
    "Energy predictions depend on the trained ML model. "
    "Charging calculations assume constant charging power."
)

# --------------------------------------------------
# 8. USER INPUTS
# --------------------------------------------------
with st.sidebar:
    st.header("EV Charging Details")

    battery = st.number_input(
        "Battery capacity (kWh)",
        min_value=5.0,
        max_value=200.0,
        value=60.0,
        step=1.0
    )

    soc = st.slider(
        "Current battery charge (%)",
        min_value=0,
        max_value=99,
        value=35
    )

    power = st.number_input(
        "Charging power (kW)",
        min_value=1.0,
        max_value=50.0,
        value=7.0,
        step=0.5
    )

    current = st.slider(
        "Current hour",
        min_value=0,
        max_value=22,
        value=16
    )

    departure = st.slider(
        "Departure hour",
        min_value=current + 1,
        max_value=24,
        value=min(24, current + 7)
    )

    st.caption(
        "Use a 24-hour clock. Departure hour 24 means midnight."
    )

# --------------------------------------------------
# 9. PREDICTION AND OPTIMIZATION
# --------------------------------------------------
if st.button("Predict and Optimize Charging", type="primary"):

    # Start with typical training values for all model features.
    row = {}

    for col in feature_columns:
        if col in numeric_features:
            row[col] = medians.get(col, 0.0)
        else:
            row[col] = modes.get(col, "Unknown")

    # Replace model features with the user's inputs
    # when the corresponding columns exist in the dataset.
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

    input_df = pd.DataFrame(
        [row],
        columns=feature_columns
    )

    try:
        transformed = preprocessor.transform(input_df)
        prediction = float(model.predict(transformed)[0])
    except Exception as error:
        st.error(
            "Prediction failed. Please check the model bundle "
            "and its library versions."
        )
        st.exception(error)
        st.stop()

    predicted_energy = max(0.0, prediction)

    # Physical limit: do not exceed the remaining battery capacity.
    remaining_capacity = battery * (100 - soc) / 100
    energy = min(predicted_energy, remaining_capacity)

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Predicted session energy",
        f"{predicted_energy:.2f} kWh"
    )

    c2.metric(
        "Energy used for scheduling",
        f"{energy:.2f} kWh"
    )

    c3.metric(
        "Remaining battery capacity",
        f"{remaining_capacity:.2f} kWh"
    )

    # --------------------------------------------------
    # 10. SCHEDULE CHARGING
    # --------------------------------------------------
    if energy <= 0:
        st.success("No additional charging energy is required.")

    else:
        best, duration = find_best_slot(
            energy,
            power,
            current,
            departure
        )

        if best is None:
            available_hours = departure - current
            possible_energy = power * available_hours

            st.error(
                "The required energy cannot be delivered before "
                "your selected departure time."
            )

            st.write(
                f"Available charging time: {available_hours:.2f} hours"
            )

            st.write(
                f"Maximum energy deliverable in that time: "
                f"{possible_energy:.2f} kWh"
            )

            st.write(
                "Try a later departure, a lower energy requirement, "
                "or a higher compatible charging power."
            )

        else:
            # Cost of starting immediately.
            immediate_cost = calculate_cost(
                current,
                duration,
                power
            )

            smart_cost = best["cost"]

            # Savings cannot be negative: immediate charging is
            # itself a feasible option, so the optimizer should
            # never choose a more expensive slot.
            savings = max(0.0, immediate_cost - smart_cost)

            savings_percent = (
                savings / immediate_cost * 100
                if immediate_cost > 0
                else 0.0
            )

            # --------------------------------------------------
            # 11. DISPLAY RECOMMENDED SCHEDULE
            # --------------------------------------------------
            st.subheader("Recommended Charging Schedule")

            a, b, c = st.columns(3)

            a.metric(
                "Charging duration",
                format_duration(duration)
            )

            b.metric(
                "Recommended start",
                format_time(best["start"])
            )

            c.metric(
                "Estimated charging cost",
                f"₹{smart_cost:.2f}"
            )

            st.write(
                f"**Scheduled finish:** {format_time(best['end'])}"
            )

            if abs(best["start"] - current) < 1e-9:
                st.success(
                    "Starting now is already the lowest-cost "
                    "option within the selected time window."
                )
            elif savings > 0:
                st.success(
                    f"Recommended schedule saves approximately "
                    f"₹{savings:.2f} ({savings_percent:.1f}%) "
                    f"compared with charging immediately."
                )
            else:
                st.info(
                    "No cost difference was found between the "
                    "selected schedule and immediate charging."
                )

            # --------------------------------------------------
            # 12. COST COMPARISON
            # --------------------------------------------------
            st.subheader("Cost Comparison")

            d, e, f = st.columns(3)

            d.metric(
                "Immediate charging",
                f"₹{immediate_cost:.2f}"
            )

            e.metric(
                "Smart schedule",
                f"₹{smart_cost:.2f}"
            )

            f.metric(
                "Estimated savings",
                f"₹{savings:.2f}",
                delta=f"{savings_percent:.1f}%"
            )

            cost_chart = pd.DataFrame({
                "Method": [
                    "Immediate charging",
                    "Smart schedule"
                ],
                "Estimated Cost (₹)": [
                    immediate_cost,
                    smart_cost
                ]
            }).set_index("Method")

            st.bar_chart(cost_chart)

            # --------------------------------------------------
            # 13. 24-HOUR TARIFF VISUALIZATION
            # --------------------------------------------------
            st.subheader("Illustrative 24-Hour Electricity Tariff")

            tariff_df = pd.DataFrame({
                "Hour": list(range(24)),
                "Tariff (₹/kWh)": [
                    get_tariff(hour)
                    for hour in range(24)
                ]
            }).set_index("Hour")

            st.line_chart(tariff_df)

            # --------------------------------------------------
            # 14. CHARGING SCHEDULE TIMELINE
            # --------------------------------------------------
            st.subheader("Charging Schedule Timeline")

            timeline = pd.DataFrame({
                "Event": [
                    "Current time",
                    "Recommended charging start",
                    "Charging finish",
                    "Departure"
                ],
                "Hour": [
                    current,
                    best["start"],
                    best["end"],
                    departure
                ]
            })

            timeline["Time"] = timeline["Hour"].apply(format_time)

            st.dataframe(
                timeline[["Event", "Time"]],
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Tariffs are illustrative, not live electricity prices. "
                "Actual costs depend on your provider's tariff, charging "
                "efficiency, power limits and vehicle behavior. "
                "The optimizer searches feasible start times in "
                "15-minute increments."
            )

# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "MSc AI/ML Project | AI-Based Smart EV Charging Scheduler "
    "and Energy Optimization System"
)
