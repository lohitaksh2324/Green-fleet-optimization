# Data Sourcing & Calibration Notes

## Why synthetic, not purely real
No public dataset covers our actual problem shape: multi-vessel fleets, 5 fuel types
(including hydrogen/ammonia, which don't exist at fleet scale in the real world yet),
alt-fuel adoption scenarios, 50-200 vessels. Real data is used to *calibrate* the
synthetic generator, not replace it.

## Sources used
1. **`ship_fuel_efficiency.csv`** (Kaggle, 1440 rows: ship_id, ship_type, route, fuel_type,
   fuel_consumption, CO2, weather, engine_efficiency)
   - Used for: fuel-per-nm baselines by ship type + fuel type, noise level, weather effect.
   - Finding: within the same ship+route, fuel consumption has CV ≈ 0.44 → used as our
     noise model's scale.
   - Finding: weather effect on fuel was weak in real data (Calm 29.5 vs Stormy 28.1,
     ~5% swing) — NOT the large penalty intuition suggests. Our weather multipliers
     (1.00 / 1.04 / 1.10) reflect this rather than an assumed large effect.

2. **`Ship_Performance_Dataset.csv`** (Kaggle, 2736 rows: speed, engine power, cargo
   weight, draft, etc.)
   - Checked whether it could calibrate the Admiralty exponents directly:
     regressed log(Power) ~ log(Speed) + log(CargoWeight).
     **Result: R² = 0.0007** — no physical relationship, i.e. this dataset's numbers
     are randomly generated, not real telemetry.
   - Used only for realistic *ranges*: speed 10-25 knots, cargo 50-2000 tons, draft,
     weather category labels. Not used for any physics calibration.

3. **EU MRV 2024 "Full ERs"** (provided by team, `2024-v239-...xlsx`, 14,162 ships,
   12,957 usable after cleaning) — real, EU-verified annual fuel/CO2/distance per ship.
   - This is now the PRIMARY physics calibration source, replacing the hand-tuned
     constant. Extracted real mean "Fuel consumption per distance [kg/n mile]" per
     ship type, matched directly onto our 5 SIH ship types:
       Bulk carrier 171.5 | Container ship 6279.4 | Oil tanker 190.6 |
       Ro-ro ship 104.4 | LNG carrier 241.4   (kg/nm, real fleet means)
   - Fit a **per-ship-type** Admiralty constant K by solving
     `fuel_per_nm = K * disp_mid^(2/3) * speed_mid^2 * 3.6 / energy_density(HFO)`
     against each real baseline (see `fuel_simulator.py`, `K_ADMIRALTY_BY_TYPE`).
   - Finding: Container ships need a K ~15x higher than Bulk/Tanker/RoRo/LNG. This is
     real, not a bug — container ships genuinely run far more installed power per
     unit of displacement (speed premium, reefer loads, schedule reliability). A
     single universal K across ship types would have hidden this; per-type K
     captures it and is a legitimate modeling choice to call out in the report.
   - Validation: regenerated synthetic fleet now lands within ~10% of the real
     EU MRV mean fuel-per-nm on every ship type (RoRo: 104.0 sim vs 104.4 real).
   - Limitation: EU MRV gives annual aggregates per ship, not per-voyage speed/load
     curves — so it calibrates the *scale* of the physics constant well, but the
     within-voyage noise model still comes from the Kaggle D1 CV (0.44), since MRV
     doesn't expose voyage-level variance.

## What's hardcoded (not derived from any dataset)
- Fuel energy densities (MJ/kg) and well-to-wake CO2 factors per fuel type — static
  reference constants from IMO alternative marine fuels lifecycle GHG guidance / DNV
  alternative fuels insight report. These should be double-checked against the actual
  source documents before the report is finalized (numbers used here are ballpark).
- K_ADMIRALTY_BY_TYPE — now fit against real EU MRV data (see above), no longer hand-tuned.

## Sanity check result (after EU MRV calibration)
Generated 1000 synthetic voyages (100 vessels x 10 voyages). Fuel-per-nm by ship type
now within ~10% of real EU MRV fleet means for all 5 ship types.

## Next steps for the Prediction track
- Fuel types beyond HFO (Methanol/Hydrogen/Ammonia at fleet scale) have no real-world
  calibration source — EU MRV fleets are still overwhelmingly HFO/MDO/LNG, so those
  three fuel types' energy-density-driven fuel_kg conversion is physics-only, not
  data-calibrated. Worth stating explicitly in the report rather than implying equal
  calibration confidence across all 5 fuels.
- Move on to training XGBoost baseline on `synthetic_fleet_voyages.csv` (Day 3 task).
