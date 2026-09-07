"""
Synthetic fleet fuel-consumption generator.
Physics: Admiralty formula  Power ∝ Displacement^(2/3) * Speed^3

Calibrated against:
  - data/kaggle_1/ship_fuel_efficiency.csv  -> fuel-per-nm baselines, noise CV, weather effect
  - data/kaggle_2/Ship_Performance_Dataset.csv -> realistic ranges for speed/cargo/draft
    (NOT used for physics calibration: log(Power) vs log(Speed)/log(Cargo) gave R^2 ~ 0.0007,
     i.e. that dataset is not physically grounded — ranges only.)

Energy density / emission factors are static reference constants (IMO/DNV alt-fuel guidance),
not derived from the Kaggle files.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---- Ship type profiles: (displacement_tons range, speed_knots range) ----
SHIP_TYPES = {
    "Bulk Carrier":    {"disp": (20_000, 180_000), "speed": (10, 15)},
    "Container Ship":  {"disp": (30_000, 220_000), "speed": (14, 24)},
    "Tanker":          {"disp": (25_000, 300_000), "speed": (10, 16)},
    "RoRo":            {"disp": (10_000, 40_000),  "speed": (14, 22)},
    "LNG Carrier":     {"disp": (60_000, 130_000), "speed": (14, 20)},
}

# ---- Fuel types: energy density (MJ/kg), well-to-wake CO2eq (kg/kg fuel), relative cost ----
# Reference figures (IMO alternative marine fuels lifecycle GHG guidelines / DNV insight report ballpark).
FUEL_TYPES = {
    "HFO":      {"energy_density": 40.0, "co2_per_kg": 3.11, "cost_per_ton": 550},
    "MDO":      {"energy_density": 42.7, "co2_per_kg": 3.21, "cost_per_ton": 700},
    "LNG":      {"energy_density": 50.0, "co2_per_kg": 2.75, "cost_per_ton": 600},
    "Methanol": {"energy_density": 19.9, "co2_per_kg": 1.38, "cost_per_ton": 450},
    "Hydrogen": {"energy_density": 120.0, "co2_per_kg": 0.00, "cost_per_ton": 4000},
    "Ammonia":  {"energy_density": 18.6, "co2_per_kg": 0.00, "cost_per_ton": 900},
}

WEATHER = {
    "Calm":     1.00,
    "Moderate": 1.04,   # weak effect, matches D1's real weather signal (~5% not ~30%)
    "Stormy":   1.10,
}

# Calibrated from D1 (Kaggle): noise CV
NOISE_CV = 0.44          # matches within-ship+route CV of fuel_consumption in D1

# Calibrated from EU MRV 2024 (real, verified fleet data, ~12,900 ships) via least-squares
# fit of fuel_per_nm = K * disp_mid^(2/3) * speed_mid^2 * 3.6 / energy_density(HFO)
# against each ship type's REAL mean fuel-per-nm [kg/nm]:
#   Bulk Carrier 171.5, Container Ship 6279.4, Tanker(Oil tanker) 190.6,
#   RoRo(Ro-ro ship) 104.4, LNG Carrier 241.4
# Container ships come out ~15x higher K than the rest — real data shows they run far more
# installed power per unit of displacement than bulk/tanker/RoRo/LNG (speed premium, reefer
# load, schedule reliability). A single universal K would have hidden this; per-type K
# captures it. See docs/data_calibration_notes.md for the derivation.
K_ADMIRALTY_BY_TYPE = {
    "Bulk Carrier":    0.00566,
    "Container Ship":  0.07731,
    "Tanker":          0.00421,
    "RoRo":            0.00419,
    "LNG Carrier":     0.00446,
}


def admiralty_power_kw(displacement_tons: float, speed_knots: float, ship_type: str) -> float:
    """Shaft power estimate from the Admiralty formula, with a per-ship-type constant
    fit against real EU MRV 2024 fuel-per-nm data."""
    k = K_ADMIRALTY_BY_TYPE[ship_type]
    return k * (displacement_tons ** (2 / 3)) * (speed_knots ** 3)


def generate_voyage(vessel_id, ship_type, fuel_type, rng=RNG):
    profile = SHIP_TYPES[ship_type]
    fuel = FUEL_TYPES[fuel_type]

    displacement = rng.uniform(*profile["disp"])
    speed = rng.uniform(*profile["speed"])
    distance_nm = rng.uniform(50, 500)
    weather = rng.choice(list(WEATHER.keys()), p=[0.5, 0.35, 0.15])
    cargo_load_pct = rng.uniform(0.4, 1.0)  # partial loads shift effective displacement a bit

    effective_disp = displacement * (0.7 + 0.3 * cargo_load_pct)
    power_kw = admiralty_power_kw(effective_disp, speed, ship_type) * WEATHER[weather]

    # power -> fuel mass via energy density (simple constant specific fuel consumption assumption)
    voyage_hours = distance_nm / speed
    energy_required_mj = power_kw * voyage_hours * 3.6  # kWh -> MJ
    fuel_kg = energy_required_mj / fuel["energy_density"]

    # calibrated multiplicative noise (weather-independent load/engine variability)
    fuel_kg *= rng.lognormal(mean=0, sigma=NOISE_CV * 0.6)

    co2_kg = fuel_kg * fuel["co2_per_kg"]
    cost_usd = (fuel_kg / 1000) * fuel["cost_per_ton"]

    return {
        "vessel_id": vessel_id,
        "ship_type": ship_type,
        "fuel_type": fuel_type,
        "displacement_tons": round(displacement, 1),
        "cargo_load_pct": round(cargo_load_pct, 3),
        "speed_knots": round(speed, 2),
        "distance_nm": round(distance_nm, 1),
        "weather": weather,
        "power_kw": round(power_kw, 1),
        "voyage_hours": round(voyage_hours, 2),
        "fuel_kg": round(fuel_kg, 2),
        "co2_kg": round(co2_kg, 2),
        "cost_usd": round(cost_usd, 2),
    }


def generate_fleet_dataset(n_vessels=100, voyages_per_vessel=10, fuel_mix=None, seed=42):
    """fuel_mix: dict of {fuel_type: probability}. Defaults to diesel-heavy baseline scenario."""
    rng = np.random.default_rng(seed)
    if fuel_mix is None:
        fuel_mix = {"HFO": 0.55, "MDO": 0.35, "LNG": 0.10}

    fuels, probs = zip(*fuel_mix.items())
    ship_type_list = list(SHIP_TYPES.keys())

    rows = []
    for v in range(n_vessels):
        vessel_id = f"V{v:04d}"
        ship_type = rng.choice(ship_type_list)
        fuel_type = rng.choice(fuels, p=probs)
        for _ in range(voyages_per_vessel):
            rows.append(generate_voyage(vessel_id, ship_type, fuel_type, rng))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_fleet_dataset(n_vessels=100, voyages_per_vessel=10)
    df.to_csv("../data/synthetic_fleet_voyages.csv", index=False)
    print(df.shape)
    print(df.groupby("ship_type")["fuel_kg"].mean())
    print("\nFuel-per-nm by ship_type (compare to D1 baselines):")
    df["fuel_per_nm"] = df.fuel_kg / df.distance_nm
    print(df.groupby("ship_type")["fuel_per_nm"].mean())
