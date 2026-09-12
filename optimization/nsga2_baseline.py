"""
NSGA-II baseline — multi-objective version (cost vs CO2), same toy fleet/physics
as classical_ga_baseline.py, for the Day 3 "classical comparator" requirement.
"""

import numpy as np
import sys
sys.path.insert(0, "../prediction")
from fuel_simulator import SHIP_TYPES, FUEL_TYPES
from classical_ga_baseline import (
    FLEET, WEATHER_FACTOR, MAX_TOTAL_HOURS, voyage_fuel_and_cost,
    CAPPED_FUELS, max_allowed_count, CII_REDUCTION_TARGET, vessel_naive_co2_per_nm,
)

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


class ToyFleetProblemMulti(ElementwiseProblem):
    """Two objectives: total cost, total CO2 (genuinely competing — cheaper fuels
    like HFO are NOT the low-CO2 ones), one constraint: schedule reliability."""

    def __init__(self):
        variables = {}
        for i, vessel in enumerate(FLEET):
            lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
            variables[f"speed_{i}"] = Real(bounds=(lo, hi))
            variables[f"fuel_{i}"] = Choice(options=list(FUEL_TYPES.keys()))
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=2 + len(CAPPED_FUELS))

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

        out["F"] = [total_cost, total_co2]
        g_fuel = [fuel_counts[f] - max_allowed_count(f, len(FLEET)) for f in CAPPED_FUELS]
        out["G"] = [total_hours - MAX_TOTAL_HOURS, cii_violation] + g_fuel


def run_nsga2(pop_size=40, n_gen=60, seed=1):
    problem = ToyFleetProblemMulti()
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=MixedVariableSampling(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    termination = get_termination("n_gen", n_gen)
    res = minimize(problem, algorithm, termination, seed=seed, verbose=False, save_history=True)

    # track hypervolume-ish proxy: best (min) cost and best (min) co2 seen so far per gen
    best_cost_curve, best_co2_curve = [], []
    for gen in res.history:
        F = gen.opt.get("F")
        best_cost_curve.append(F[:, 0].min())
        best_co2_curve.append(F[:, 1].min())

    return res, best_cost_curve, best_co2_curve


if __name__ == "__main__":
    res, cost_curve, co2_curve = run_nsga2()
    print("=== NSGA-II baseline (toy 8-vessel fleet, cost vs CO2) ===")
    print(f"Pareto front size: {len(res.F)}")
    print(f"Cost range on front:  {res.F[:,0].min():.0f} - {res.F[:,0].max():.0f} USD")
    print(f"CO2 range on front:   {res.F[:,1].min():.0f} - {res.F[:,1].max():.0f} kg")

    # pick the "balanced" knee point: min normalized sum
    norm = (res.F - res.F.min(axis=0)) / (res.F.max(axis=0) - res.F.min(axis=0) + 1e-9)
    knee_idx = np.argmin(norm.sum(axis=1))
    print(f"\nBalanced (knee-point) solution: cost={res.F[knee_idx,0]:.0f} USD, CO2={res.F[knee_idx,1]:.0f} kg")

    np.save("../data/nsga2_pareto_front.npy", res.F)
    np.save("../data/nsga2_cost_convergence.npy", np.array(cost_curve))
    np.save("../data/nsga2_co2_convergence.npy", np.array(co2_curve))
    print("\nPareto front + convergence curves saved for comparison.")
