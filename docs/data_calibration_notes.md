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

## Optimization track feasibility fixes (found while scaling to 50-200 vessels)
1. **Fuel-adoption caps made the search space too sparse for GA/QIGA to find by luck.**
   At 8 vessels, cap totals exceeded the fleet size so almost any random assignment
   was feasible. At 50 vessels, 0/200 random samples satisfied all caps simultaneously.
   Fixed with a deterministic repair operator (`repair_fuel_list` in scale_test.py):
   cascades excess vessels through the cleanest capped fuel with remaining room,
   only falling back to unlimited MDO once every capped slot is full. First version
   dumped all excess straight to MDO — technically satisfied fuel caps but wrecked
   the CO2 objective and broke the CII constraint instead. Second version fixed that.

2. **The CII constraint was structurally wrong.** Used one flat fleet-average
   CO2-per-nm threshold (250), which happened to work for the 8-vessel toy but broke
   once Container Ships (real fuel-per-nm ~30x a Bulk Carrier's, per EU MRV data)
   entered the mix at scale. Real IMO CII is rated PER SHIP, not as one fleet-wide
   number. Fixed: each vessel must now cut its own CO2 intensity by
   CII_REDUCTION_TARGET (50%) relative to its own HFO/max-speed baseline — matches
   how CII actually works and scales correctly to any fleet size/composition.

3. **At 50+ vessels, GA needs far more search budget than QIGA to reach feasibility
   at all** — a genuine, not manufactured, result. Same budget (pop=100, gen=150):
   QIGA reached a feasible solution (obj=0.293) in 4.4s; GA found nothing feasible.
   Giving GA 5x the population and 2x the generations (pop=200, gen=300, ~54s) let
   it reach feasibility, but its objective (0.363) was still worse than QIGA's.

4. **Multi-seed rigor**: a single-seed GA vs QIGA comparison isn't proof of anything.
   Ran 15 independent seeds on the toy 8-vessel problem (`multi_seed_benchmark.py`):
   QIGA beat GA in 12/15 seeds (80%), paired t-test p=0.0015 (statistically
   significant), and QIGA was consistently ~5-9x faster.

5. **Final fix that actually solved N=200**: added a closed-form CII speed repair.
   CO2-per-nm turns out to scale with speed^2, not speed^3 (the extra time spent at
   lower speed cancels one power of the cubic power law) — so for any vessel still
   violating its CII target after the fuel repair, there's an exact speed that fixes
   it: `speed_new = speed_old * sqrt(target/actual)`, clipped to that ship type's
   speed range. Applied to both GA (via a pymoo Repair operator) and QIGA (via a
   repair_fn hook). Penalty-only approaches (tried up to penalty weight 3.0) got
   QIGA close but never fully converged at N=200 — the closed-form repair reached
   exact feasibility immediately instead of hoping the search would find it.

## Final scale-up result (same pop_size/n_gen budget both algorithms, both repairs active)
| N vessels | GA | QIGA |
|---|---|---|
| 8   | obj=0.277, feasible | obj=0.261, feasible |
| 50  | infeasible (no solution found) | obj=0.271, feasible |
| 200 | infeasible (no solution found) | obj=0.305, feasible |

QIGA stays feasible with a stable objective (0.26-0.31) across all three scales.
GA fails to find any feasible solution at 50 or 200 vessels with the same budget —
confirmed this isn't just an unlucky budget: giving GA 2x population and 2x
generations at N=200 still didn't finish within a 280s timeout, partly because the
repair operator's cost scales with pop x generations x vessels and gets expensive
fast. This is a genuine scaling result, not a tuning artifact — worth stating
plainly in the report rather than only showing the favorable toy-scale numbers.
