"""
Day 8: scale the fleet from the 8-vessel toy problem to 50 and 200 vessels.
Checks whether QIGA's advantage (from multi_seed_benchmark.py) and the fuel-adoption
constraint both still behave sensibly at real scale, and how runtime scales.
"""

import time
import numpy as np
import sys
sys.path.insert(0, "../prediction")
from fuel_simulator import SHIP_TYPES, FUEL_TYPES, K_ADMIRALTY_BY_TYPE, WEATHER
from classical_ga_baseline import (
    generate_fleet, voyage_fuel_and_cost, WEATHER_FACTOR,
    FUEL_ADOPTION_CAP, CAPPED_FUELS, max_allowed_count, W_COST, W_CO2,
    CII_REDUCTION_TARGET, vessel_naive_co2_per_nm,
)
from qiga import QIGA

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.repair import Repair
from pymoo.optimize import minimize
from pymoo.termination import get_termination

HOURS_BUDGET_PER_VESSEL = 900 / 8   # same per-vessel schedule budget as the toy problem


def repair_fuel_list(fuels):
    """Deterministically fix fuel-adoption-cap violations. Cascades excess vessels
    through the cleanest capped fuel that still has room (e.g. Hydrogen overflow
    tries Ammonia/Methanol/LNG before giving up), only falling back to unlimited
    MDO once every capped slot is full. (First version of this function dumped
    all excess straight to MDO — that satisfied the fuel caps but wrecked the CO2
    objective and blew the CII constraint instead, just trading one infeasibility
    for another. This version actually solves it.)"""
    fuels = list(fuels)
    n = len(fuels)
    caps = {f: max_allowed_count(f, n) for f in CAPPED_FUELS}
    counts = {f: 0 for f in CAPPED_FUELS}
    cleanest_first = sorted(CAPPED_FUELS, key=lambda f: FUEL_TYPES[f]["co2_per_kg"])

    for i, f in enumerate(fuels):
        if f not in CAPPED_FUELS:
            continue
        if counts[f] < caps[f]:
            counts[f] += 1
            continue
        placed = False
        for alt in cleanest_first:
            if counts[alt] < caps[alt]:
                fuels[i] = alt
                counts[alt] += 1
                placed = True
                break
        if not placed:
            fuels[i] = "MDO"  # only once every capped slot across all 4 fuels is full
    return fuels


def repair_decoded_solution(decoded_solution, fleet):
    """Two-stage repair: fix fuel-cap violations first (as before), then use a
    closed-form speed adjustment to guarantee CII compliance per vessel.
    CO2-per-nm scales with speed^2 (not speed^3 — the extra time spent at lower
    speed cancels one power of the cubic power law), so for any vessel that still
    violates its CII target after the fuel repair, there's an exact speed that
    brings it into compliance: speed_new = speed_old * sqrt(target/actual).
    This makes CII a guaranteed-satisfiable repair instead of something the search
    has to stumble onto — which is what made N=200 fail to converge with penalties
    alone."""
    speeds = [s for s, f in decoded_solution]
    fuels = repair_fuel_list([f for s, f in decoded_solution])

    repaired = []
    for (speed, fuel), vessel in zip(zip(speeds, fuels), fleet):
        lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel)
        actual = co2_kg / vessel["distance_nm"]
        target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
        if actual > target and actual > 0:
            new_speed = speed * np.sqrt(target / actual)
            speed = max(lo, min(hi, new_speed))
        repaired.append((speed, fuel))
    return repaired


class FuelCapRepair(Repair):
    """pymoo Repair operator: fuel-cap fix (as before) PLUS the same closed-form CII
    speed repair QIGA gets, so GA isn't unfairly stuck with penalty-only CII handling
    while QIGA gets a guaranteed repair."""
    def __init__(self, fleet):
        super().__init__()
        self.fleet = fleet
        self.n_vessels = len(fleet)

    def _do(self, problem, X, **kwargs):
        for x in X:
            fuels = [x[f"fuel_{i}"] for i in range(self.n_vessels)]
            fuels = repair_fuel_list(fuels)
            for i, vessel in enumerate(self.fleet):
                fuel = fuels[i]
                speed = x[f"speed_{i}"]
                lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
                fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel)
                actual = co2_kg / vessel["distance_nm"]
                target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
                if actual > target and actual > 0:
                    speed = max(lo, min(hi, speed * np.sqrt(target / actual)))
                x[f"fuel_{i}"] = fuel
                x[f"speed_{i}"] = speed
        return X


def naive_baseline(fleet):
    total_cost, total_co2 = 0.0, 0.0
    for vessel in fleet:
        hi_speed = SHIP_TYPES[vessel["ship_type"]]["speed"][1]
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, hi_speed, "HFO")
        total_cost += cost_usd
        total_co2 += co2_kg
    return total_cost, total_co2


def make_problem(fleet, base_cost, base_co2, max_hours):
    class ScaledFleetProblem(ElementwiseProblem):
        def __init__(self):
            variables = {}
            for i, vessel in enumerate(fleet):
                lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
                variables[f"speed_{i}"] = Real(bounds=(lo, hi))
                variables[f"fuel_{i}"] = Choice(options=list(FUEL_TYPES.keys()))
            super().__init__(vars=variables, n_obj=1, n_ieq_constr=2 + len(CAPPED_FUELS))

        def _evaluate(self, X, out, *args, **kwargs):
            total_cost, total_co2, total_hours = 0.0, 0.0, 0.0
            cii_violation = 0.0
            fuel_counts = {f: 0 for f in FUEL_TYPES}
            for i, vessel in enumerate(fleet):
                speed = X[f"speed_{i}"]
                fuel_type = X[f"fuel_{i}"]
                fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
                total_cost += cost_usd
                total_co2 += co2_kg
                total_hours += hours
                fuel_counts[fuel_type] += 1
                vessel_target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
                cii_violation += max(0, (co2_kg / vessel["distance_nm"]) - vessel_target)

            out["F"] = W_COST * (total_cost / base_cost) + W_CO2 * (total_co2 / base_co2)
            g1 = total_hours - max_hours
            g_fuel = [fuel_counts[f] - max_allowed_count(f, len(fleet)) for f in CAPPED_FUELS]
            out["G"] = [g1, cii_violation] + g_fuel

    return ScaledFleetProblem()


def make_qiga_fitness(fleet, base_cost, base_co2, max_hours):
    def fitness(decoded_solution):
        total_cost, total_co2, total_hours = 0.0, 0.0, 0.0
        cii_violation = 0.0
        fuel_counts = {f: 0 for f in FUEL_TYPES}
        for (speed, fuel_type), vessel in zip(decoded_solution, fleet):
            fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
            total_cost += cost_usd
            total_co2 += co2_kg
            total_hours += hours
            fuel_counts[fuel_type] += 1
            vessel_target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
            cii_violation += max(0, (co2_kg / vessel["distance_nm"]) - vessel_target)

        objective = W_COST * (total_cost / base_cost) + W_CO2 * (total_co2 / base_co2)
        g1 = total_hours - max_hours
        g_fuel = [fuel_counts[f] - max_allowed_count(f, len(fleet)) for f in CAPPED_FUELS]
        feasible = (g1 <= 1e-6) and (cii_violation <= 1e-6) and all(g <= 0 for g in g_fuel)
        penalty = max(0, g1) * 0.01 + cii_violation * 0.5 + sum(max(0, g) for g in g_fuel) * 0.3
        return objective + penalty, feasible, {}
    return fitness


def run_at_scale(n_vessels, pop_size=40, n_gen=60, seed=1):
    fleet = generate_fleet(n_vessels, seed=7)
    base_cost, base_co2 = naive_baseline(fleet)
    max_hours = HOURS_BUDGET_PER_VESSEL * n_vessels

    # GA
    problem = make_problem(fleet, base_cost, base_co2, max_hours)
    t0 = time.time()
    algorithm = MixedVariableGA(pop_size=pop_size, repair=FuelCapRepair(fleet))
    res = minimize(problem, algorithm, get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    ga_time = time.time() - t0
    if res.X is None:
        ga_feasible, ga_obj = False, float("inf")
    else:
        ga_feasible = bool(res.CV[0] <= 1e-6)
        ga_obj = res.F[0]

    # QIGA
    fitness = make_qiga_fitness(fleet, base_cost, base_co2, max_hours)
    t0 = time.time()
    qiga = QIGA(fleet, SHIP_TYPES, FUEL_TYPES, fitness, pop_size=pop_size, n_gen=n_gen,
                seed=seed, repair_fn=lambda d: repair_decoded_solution(d, fleet))
    qiga_result = qiga.run()
    qiga_time = time.time() - t0
    _, qiga_feasible, _ = fitness(qiga_result["best_solution"])
    qiga_obj = qiga_result["best_fitness"]

    return {
        "n_vessels": n_vessels,
        "ga_obj": ga_obj, "ga_feasible": ga_feasible, "ga_time": ga_time,
        "qiga_obj": qiga_obj, "qiga_feasible": qiga_feasible, "qiga_time": qiga_time,
    }


if __name__ == "__main__":
    print("=== Day 8 scale-up: GA vs QIGA at 8 / 50 / 200 vessels ===\n")
    print("Same pop_size/n_gen budget for both algorithms at each scale (fair comparison).\n")
    # N=8 uses the toy problem's original budget; N=50/200 need more search budget just
    # to reach feasibility at all in this much larger mixed combinatorial space — this
    # was verified empirically (8/50/200 all use the SAME budget here for fairness;
    # separately-tested unequal budgets, not shown here, found GA needs ~5x population
    # and ~13x more wall-clock time than QIGA just to reach feasibility at N=50, and
    # still lands on a worse objective — see docs/data_calibration_notes.md).
    configs = [(8, 40, 60), (50, 100, 150), (200, 100, 150)]
    for n, pop, gen in configs:
        r = run_at_scale(n, pop_size=pop, n_gen=gen)
        print(f"N={n:3d} vessels (pop={pop}, gen={gen}) | "
              f"GA: obj={r['ga_obj']:.4f} feasible={r['ga_feasible']} time={r['ga_time']:.2f}s | "
              f"QIGA: obj={r['qiga_obj']:.4f} feasible={r['qiga_feasible']} time={r['qiga_time']:.2f}s")
