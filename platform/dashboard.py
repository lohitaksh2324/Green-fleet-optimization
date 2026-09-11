"""
Deliverable 4: Streamlit dashboard. Calls the FastAPI backend (api.py) so the
dashboard and any other client share the exact same optimizer/predictor logic —
no duplicated fitness functions between the UI and the API.

Run:
    1. Start the backend:  cd platform && uvicorn api:app --reload --port 8000
    2. Start the dashboard: cd platform && streamlit run dashboard.py
"""

import streamlit as st
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Egreen Quanta", layout="wide")
st.title("Egreen Quanta — Fleet Fuel &amp; Emissions Dashboard")

tab_predict, tab_optimize, tab_scenarios, tab_report = st.tabs(
    ["Fuel prediction", "Fleet optimizer", "Scenario comparison", "PDF report"]
)

# ---------------- Tab 1: Prediction ----------------
with tab_predict:
    st.subheader("Predict fuel consumption for a single voyage")
    col1, col2 = st.columns(2)
    with col1:
        ship_type = st.selectbox("Ship type", ["Bulk Carrier", "Container Ship", "Tanker", "RoRo", "LNG Carrier"])
        fuel_type = st.selectbox("Fuel type", ["HFO", "MDO", "LNG", "Methanol", "Hydrogen", "Ammonia"])
        weather = st.selectbox("Weather", ["Calm", "Moderate", "Stormy"])
    with col2:
        displacement = st.number_input("Displacement (tons)", value=100000.0)
        speed = st.number_input("Speed (knots)", value=14.0)
        distance = st.number_input("Distance (nm)", value=500.0)
        cargo_load = st.slider("Cargo load %", 0.0, 1.0, 0.8)

    if st.button("Predict fuel consumption"):
        try:
            resp = requests.post(f"{API_BASE}/predict", json={
                "ship_type": ship_type, "fuel_type": fuel_type, "weather": weather,
                "displacement_tons": displacement, "cargo_load_pct": cargo_load,
                "speed_knots": speed, "distance_nm": distance,
            }, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            st.metric("Predicted fuel consumption", f"{result['predicted_fuel_kg']:,.0f} kg")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API backend. Start it first: `uvicorn api:app --reload --port 8000`")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------------- Tab 2: Optimizer ----------------
with tab_optimize:
    st.subheader("Run the fleet optimizer (QIGA)")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_vessels = st.number_input("Number of vessels", min_value=1, max_value=1440, value=8)
    with col2:
        pop_size = st.number_input("Population size", min_value=10, value=40)
    with col3:
        n_gen = st.number_input("Generations", min_value=10, value=60)

    st.caption(
        "Larger fleets need a bigger population/generation budget to converge — "
        "tested up to 1440 vessels at pop=100/gen=150 (~170s). Use small values here "
        "for a fast interactive preview."
    )

    if st.button("Run optimization"):
        with st.spinner(f"Running QIGA on {n_vessels} vessels..."):
            try:
                resp = requests.post(f"{API_BASE}/optimize", json={
                    "n_vessels": int(n_vessels), "pop_size": int(pop_size),
                    "n_gen": int(n_gen), "seed": 1,
                }, timeout=300)
                resp.raise_for_status()
                result = resp.json()

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Objective", f"{result['objective']:.4f}")
                c2.metric("Feasible", "Yes" if result["feasible"] else "No")
                c3.metric("Total cost", f"${result['total_cost_usd']:,.0f}")
                c4.metric("Total CO2", f"{result['total_co2_kg']:,.0f} kg")
                st.caption(f"Runtime: {result['runtime_seconds']:.2f}s")

                st.subheader("Fleet allocation")
                df = pd.DataFrame(result["allocation"])
                st.dataframe(df, use_container_width=True)

                st.subheader("Fuel mix")
                st.bar_chart(df["fuel_type"].value_counts())
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API backend. Start it first: `uvicorn api:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------- Tab 3: Scenarios ----------------
with tab_scenarios:
    st.subheader("Compare fuel-adoption scenarios")
    st.caption("Each scenario caps how much of the fleet can use each fuel type, "
               "reflecting different assumptions about bunkering infrastructure.")

    n_vessels_s = st.number_input("Number of vessels ", min_value=1, max_value=1440, value=8, key="scenario_n")
    scenarios_to_run = st.multiselect(
        "Scenarios to compare",
        ["diesel_only", "lng_mix_30pct", "current_baseline", "aggressive_hydrogen"],
        default=["diesel_only", "current_baseline", "aggressive_hydrogen"],
    )

    if st.button("Run scenario comparison"):
        rows = []
        progress = st.progress(0, text="Running scenarios...")
        for i, scenario in enumerate(scenarios_to_run):
            try:
                resp = requests.post(
                    f"{API_BASE}/optimize/scenario/{scenario}",
                    json={"n_vessels": int(n_vessels_s), "pop_size": 40, "n_gen": 60, "seed": 1},
                    timeout=300,
                )
                resp.raise_for_status()
                r = resp.json()
                rows.append({
                    "scenario": scenario,
                    "cost_usd": r["total_cost_usd"],
                    "co2_kg": r["total_co2_kg"],
                    "feasible": r["feasible"],
                })
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API backend. Start it first: `uvicorn api:app --reload --port 8000`")
                break
            except Exception as e:
                st.error(f"Scenario '{scenario}' failed: {e}")
            progress.progress((i + 1) / max(len(scenarios_to_run), 1))

        if rows:
            df = pd.DataFrame(rows).set_index("scenario")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cost by scenario")
                st.bar_chart(df["cost_usd"])
            with col2:
                st.subheader("CO2 by scenario")
                st.bar_chart(df["co2_kg"])
            st.dataframe(df, use_container_width=True)

# ---------------- Tab 4: PDF report ----------------
with tab_report:
    st.subheader("Generate the technical report")
    st.caption(
        "Builds a 6-page PDF covering the prediction model, GA/QIGA/NSGA-II "
        "benchmark, multi-seed statistical validation, scale-up results, and "
        "scenario comparison — pulled from the project's saved experiment data. "
        "Takes about 15-20 seconds (runs a live QIGA pass for the scenario chart)."
    )

    if st.button("Generate PDF report"):
        with st.spinner("Building report..."):
            try:
                resp = requests.post(f"{API_BASE}/report/generate", timeout=90)
                resp.raise_for_status()
                st.success("Report generated.")
                st.download_button(
                    label="Download PDF",
                    data=resp.content,
                    file_name="Egreen_Quanta_Technical_Report.pdf",
                    mime="application/pdf",
                )
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API backend. Start it first: `uvicorn api:app --reload --port 8000`")
            except Exception as e:
                st.error(f"Report generation failed: {e}")
