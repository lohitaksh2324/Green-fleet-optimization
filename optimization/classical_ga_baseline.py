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

WEATHER_FACTOR = WEATHER["Moderate"]  # fixed average condition for the toy problem
MAX_TOTAL_HOURS = 900   # schedule-reliability budget across the whole fleet
CII_THRESHOLD_KG_PER_NM = 250  # fleet-average compliance cap (toy value)

W_COST = 0.5   # objective weight: cost
W_CO2 = 0.5    # objective weight: emissions (kept equal-weighted for the toy baseline)


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


class ToyFleetProblem(ElementwiseProblem):
    def __init__(self):
        variables = {}
        for i, vessel in enumerate(FLEET):
            lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
            variables[f"speed_{i}"] = Real(bounds=(lo, hi))
            variables[f"fuel_{i}"] = Choice(options=list(FUEL_TYPES.keys()))
        super().__init__(vars=variables, n_obj=1, n_ieq_constr=2)

    def _evaluate(self, X, out, *args, **kwargs):
        total_cost = 0.0
        total_co2 = 0.0
        total_hours = 0.0
        total_distance = 0.0
        for i, vessel in enumerate(FLEET):
            speed = X[f"speed_{i}"]
            fuel_type = X[f"fuel_{i}"]
            fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
            total_cost += cost_usd
            total_co2 += co2_kg
            total_hours += hours
            total_distance += vessel["distance_nm"]

        # normalize cost/co2 to comparable scale before weighting (toy normalization constants)
        norm_cost = total_cost / 50000
        norm_co2 = total_co2 / 500000
        out["F"] = W_COST * norm_cost + W_CO2 * norm_co2

        g1 = total_hours - MAX_TOTAL_HOURS                      # schedule reliability
        g2 = (total_co2 / total_distance) - CII_THRESHOLD_KG_PER_NM  # CII-style compliance
        out["G"] = [g1, g2]


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
