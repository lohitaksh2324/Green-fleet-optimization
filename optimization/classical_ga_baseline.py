"""
Deliverable 2 (math formulation, toy realization) + Day 3 classical GA baseline.

Decision variables per vessel: speed (continuous), fuel_type (categorical).
Objective: minimize weighted(total fuel cost, total CO2 emissions).
Constraints: schedule reliability (total voyage time <= budget),
             CII-style compliance (fleet-average CO2 per distance <= threshold).

This toy problem (5-10 vessels) is the baseline QIGA (Day 4-6) must match/beat.
Reuses the SAME physics as prediction/fuel_simulator.py so optimizer and predictor
agree on what "fuel cost" means.
"""

import numpy as np
import sys
sys.path.insert(0, "../prediction")
from fuel_simulator import SHIP_TYPES, FUEL_TYPES, K_ADMIRALTY_BY_TYPE, WEATHER

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- Toy fleet setup: 8 vessels, fixed type + route, deciding speed & fuel per vessel ---
N_VESSELS = 8
rng = np.random.default_rng(7)
ship_type_list = list(SHIP_TYPES.keys())

FLEET = []
for i in range(N_VESSELS):
    st = ship_type_list[i % len(ship_type_list)]
    disp = np.mean(SHIP_TYPES[st]["disp"])  # fixed mid-size vessel per slot
    distance = rng.uniform(200, 800)         # fixed route length (nm)
    FLEET.append({"ship_type": st, "displacement": disp, "distance_nm": distance})


def generate_fleet(n_vessels, seed=7):
    """Reusable fleet generator for scale-up testing (Day 8) — same structure as the
    module-level toy FLEET above, parameterized so it can be called at 50-200 vessels."""
    r = np.random.default_rng(seed)
    fleet = []
    for i in range(n_vessels):
        st = ship_type_list[i % len(ship_type_list)]
        disp = np.mean(SHIP_TYPES[st]["disp"])
        distance = r.uniform(200, 800)
        fleet.append({"ship_type": st, "displacement": disp, "distance_nm": distance})
    return fleet

WEATHER_FACTOR = WEATHER["Moderate"]  # fixed average condition for the toy problem
MAX_TOTAL_HOURS = 900   # schedule-reliability budget across the whole fleet
CII_THRESHOLD_KG_PER_NM = 250  # fleet-average compliance cap (toy value)

W_COST = 0.3   # objective weight: cost — one of three stated objectives, not the priority
W_CO2 = 0.7    # objective weight: emissions — PDF's Background/Description frame this as
               # the primary "green fleet" goal; cost matters but shouldn't dominate it


def voyage_fuel_and_cost(vessel, speed, fuel_type):
    st = vessel["ship_type"]
    k = K_ADMIRALTY_BY_TYPE[st]
    power_kw = k * (vessel["displacement"] ** (2 / 3)) * (speed ** 3) * WEATHER_FACTOR
    hours = vessel["distance_nm"] / speed
    energy_mj = power_kw * hours * 3.6
    fuel = FUEL_TYPES[fuel_type]
    fuel_kg = energy_mj / fuel["energy_density"]
    co2_kg = fuel_kg * fuel["co2_per_kg"]
    cost_usd = (fuel_kg / 1000) * fuel["cost_per_ton"]
    return fuel_kg, co2_kg, cost_usd, hours


def naive_baseline_totals():
    """Max-speed, all-HFO reference point — used to normalize cost/CO2 onto a comparable
    relative scale, so a 1% cost improvement and a 1% CO2 improvement count equally
    before the W_COST/W_CO2 weights are applied. (Previously these were divided by
    arbitrary constants that happened to make cost easier to improve than CO2 — that
    silently let the optimizer trade emissions for cost. Fixed here.)"""
    total_cost, total_co2 = 0.0, 0.0
    for vessel in FLEET:
        hi_speed = SHIP_TYPES[vessel["ship_type"]]["speed"][1]
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, hi_speed, "HFO")
        total_cost += cost_usd
        total_co2 += co2_kg
    return total_cost, total_co2


BASE_COST, BASE_CO2 = naive_baseline_totals()

# --- Fuel adoption-rate constraint (feasibility fix) ---
# Without this, the optimizer happily converges on a 100% Hydrogen/Ammonia fleet,
# which is mathematically optimal in the model but not deployable: bunkering
# infrastructure for these fuels barely exists yet. Caps reflect rough current/
# near-term real-world readiness, not physics — tune per scenario (Day 10 of the
# plan: diesel-only / 30% LNG mix / aggressive hydrogen adoption all map to
# different values here).
import math

FUEL_ADOPTION_CAP = {
    "HFO": 1.0,
    "MDO": 1.0,
    "LNG": 0.50,       # established bunkering at major ports
    "Methanol": 0.30,  # small but growing number of bunkering points
    "Hydrogen": 0.15,  # early-stage, minimal infrastructure
    "Ammonia": 0.10,   # earliest stage, largely pilot projects
}
CAPPED_FUELS = [f for f, cap in FUEL_ADOPTION_CAP.items() if cap < 1.0]


def max_allowed_count(fuel_type, n_vessels):
    return math.ceil(FUEL_ADOPTION_CAP[fuel_type] * n_vessels)


# --- CII constraint (feasibility fix #2) ---
# Original version used ONE flat fleet-average CO2-per-nm threshold (250). That
# happened to work for the 8-vessel toy but is structurally wrong: real IMO CII is
# rated PER SHIP (A-E band on that ship's own carbon intensity), not a single number
# averaged across a mixed fleet. It breaks the moment Container Ships enter the mix —
# their real fuel-per-nm is ~30x a Bulk Carrier's (EU MRV: 6279 vs 171 kg/nm) — so a
# flat aggregate threshold either has zero margin for anyone once Container Ships are
# present, or was only ever satisfiable by coincidence at small scale. Fixed to match
# how CII actually works: each vessel must cut its OWN carbon intensity by a fixed
# fraction relative to its own HFO/max-speed baseline.
CII_REDUCTION_TARGET = 0.5  # each vessel must be <= 50% of its own dirty baseline


def vessel_naive_co2_per_nm(vessel):
    hi_speed = SHIP_TYPES[vessel["ship_type"]]["speed"][1]
    _, co2_kg, _, _ = voyage_fuel_and_cost(vessel, hi_speed, "HFO")
    return co2_kg / vessel["distance_nm"]


class ToyFleetProblem(ElementwiseProblem):
    def __init__(self):
        variables = {}
        for i, vessel in enumerate(FLEET):
            lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
            variables[f"speed_{i}"] = Real(bounds=(lo, hi))
            variables[f"fuel_{i}"] = Choice(options=list(FUEL_TYPES.keys()))
        super().__init__(vars=variables, n_obj=1, n_ieq_constr=2 + len(CAPPED_FUELS))

    def _evaluate(self, X, out, *args, **kwargs):
        total_cost = 0.0
        total_co2 = 0.0
        total_hours = 0.0
        cii_violation = 0.0
        fuel_counts = {f: 0 for f in FUEL_TYPES}
        for i, vessel in enumerate(FLEET):
            speed = X[f"speed_{i}"]
            fuel_type = X[f"fuel_{i}"]
            fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
            total_cost += cost_usd
            total_co2 += co2_kg
            total_hours += hours
            fuel_counts[fuel_type] += 1

            vessel_target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
            cii_violation += max(0, (co2_kg / vessel["distance_nm"]) - vessel_target)

        # normalize relative to naive baseline (max speed, all HFO) so a 1% cost
        # improvement and a 1% CO2 improvement carry equal weight before W_COST/W_CO2
        norm_cost = total_cost / BASE_COST
        norm_co2 = total_co2 / BASE_CO2
        out["F"] = W_COST * norm_cost + W_CO2 * norm_co2

        g1 = total_hours - MAX_TOTAL_HOURS                      # schedule reliability
        g2 = cii_violation                                      # per-vessel CII compliance (sum of violations)
        g_fuel = [fuel_counts[f] - max_allowed_count(f, len(FLEET)) for f in CAPPED_FUELS]
        out["G"] = [g1, g2] + g_fuel


def run_ga(pop_size=40, n_gen=60, seed=1):
    problem = ToyFleetProblem()
    algorithm = MixedVariableGA(pop_size=pop_size)
    termination = get_termination("n_gen", n_gen)

    res = minimize(problem, algorithm, termination, seed=seed, verbose=False, save_history=True)

    convergence = [gen.opt.get("F")[0][0] for gen in res.history]
    return res, convergence


if __name__ == "__main__":
    res, convergence = run_ga()
    print("=== Classical GA baseline (toy 8-vessel fleet) ===")
    print(f"Best objective value: {res.F[0]:.4f}")
    print(f"Feasible: {res.CV[0] <= 0}")
    print(f"Convergence (best F per generation, every 10th gen):")
    print([round(c, 4) for c in convergence[::10]])

    print("\nBest solution:")
    for i, vessel in enumerate(FLEET):
        speed = res.X[f"speed_{i}"]
        fuel = res.X[f"fuel_{i}"]
        print(f"  Vessel {i} ({vessel['ship_type']}): speed={speed:.2f}kn, fuel={fuel}")

    np.save("../data/ga_baseline_convergence.npy", np.array(convergence))
    print("\nConvergence curve saved to data/ga_baseline_convergence.npy (for QIGA comparison)")
