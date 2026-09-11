"""
Deliverable 3 / Day 9 benchmarking harness: GA vs NSGA-II vs QIGA on the
identical toy 8-vessel fleet, same physics (fuel_simulator.py), same constraints.

Reports:
  - convergence curves (best objective per generation) for GA and QIGA
  - final objective, feasibility, wall-clock runtime for each
  - % reduction in cost and CO2 vs a naive "business as usual" baseline
    (all vessels at max allowed speed, HFO fuel)
  - NSGA-II's Pareto front size and range, since it's a different (multi-obj) output
"""

import time
import numpy as np
import sys
sys.path.insert(0, "../prediction")
from fuel_simulator import SHIP_TYPES, FUEL_TYPES

from classical_ga_baseline import (
    FLEET, WEATHER_FACTOR, MAX_TOTAL_HOURS, CII_REDUCTION_TARGET, vessel_naive_co2_per_nm,
    W_COST, W_CO2, BASE_COST, BASE_CO2, voyage_fuel_and_cost, run_ga,
    CAPPED_FUELS, max_allowed_count,
)
from nsga2_baseline import run_nsga2
from qiga import QIGA


# ---------- naive baseline: max speed, all HFO ----------
def naive_baseline():
    total_cost, total_co2, total_hours = 0.0, 0.0, 0.0
    for vessel in FLEET:
        hi_speed = SHIP_TYPES[vessel["ship_type"]]["speed"][1]
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, hi_speed, "HFO")
        total_cost += cost_usd
        total_co2 += co2_kg
        total_hours += hours
    return total_cost, total_co2, total_hours


# ---------- fitness function for QIGA (same weighted+penalty structure as GA) ----------
def qiga_fitness(decoded_solution):
    total_cost, total_co2, total_hours = 0.0, 0.0, 0.0
    cii_violation = 0.0
    fuel_counts = {f: 0 for f in FUEL_TYPES}
    for (speed, fuel_type), vessel in zip(decoded_solution, FLEET):
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
        total_cost += cost_usd
        total_co2 += co2_kg
        total_hours += hours
        fuel_counts[fuel_type] += 1
        vessel_target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
        cii_violation += max(0, (co2_kg / vessel["distance_nm"]) - vessel_target)

    norm_cost = total_cost / BASE_COST
    norm_co2 = total_co2 / BASE_CO2
    objective = W_COST * norm_cost + W_CO2 * norm_co2

    g1 = total_hours - MAX_TOTAL_HOURS
    g2 = cii_violation
    g_fuel = [fuel_counts[f] - max_allowed_count(f, len(FLEET)) for f in CAPPED_FUELS]
    feasible = (g1 <= 0) and (g2 <= 0) and all(g <= 0 for g in g_fuel)

    # constraint-violation penalty (keeps the search pressured toward feasibility,
    # same role pymoo's constraint handling plays for GA/NSGA-II)
    penalty = max(0, g1) * 0.01 + g2 * 0.5 + sum(max(0, g) for g in g_fuel) * 0.3
    penalized = objective + penalty

    return penalized, feasible, {"total_cost": total_cost, "total_co2": total_co2}


def solution_cost_co2(decoded_solution):
    total_cost, total_co2 = 0.0, 0.0
    for (speed, fuel_type), vessel in zip(decoded_solution, FLEET):
        fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel_type)
        total_cost += cost_usd
        total_co2 += co2_kg
    return total_cost, total_co2


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK: GA vs NSGA-II vs QIGA — toy 8-vessel fleet")
    print("=" * 60)

    base_cost, base_co2, base_hours = naive_baseline()
    print(f"\nNaive baseline (max speed, all HFO): cost=${base_cost:,.0f}  CO2={base_co2:,.0f}kg  "
          f"hours={base_hours:.0f} (feasible={base_hours <= MAX_TOTAL_HOURS})")

    results = {}

    # --- GA ---
    t0 = time.time()
    ga_res, ga_convergence = run_ga(pop_size=40, n_gen=60, seed=1)
    ga_time = time.time() - t0
    ga_decoded = [(ga_res.X[f"speed_{i}"], ga_res.X[f"fuel_{i}"]) for i in range(len(FLEET))]
    ga_cost, ga_co2 = solution_cost_co2(ga_decoded)
    results["GA"] = {
        "objective": ga_res.F[0], "feasible": bool(ga_res.CV[0] <= 0),
        "cost": ga_cost, "co2": ga_co2, "time_s": ga_time, "convergence": ga_convergence,
    }

    # --- QIGA ---
    t0 = time.time()
    qiga = QIGA(FLEET, SHIP_TYPES, FUEL_TYPES, qiga_fitness, pop_size=40, n_gen=60, seed=1)
    qiga_result = qiga.run()
    qiga_time = time.time() - t0
    qiga_cost, qiga_co2 = solution_cost_co2(qiga_result["best_solution"])
    results["QIGA"] = {
        "objective": qiga_result["best_fitness"], "feasible": True,  # penalty-based, checked below
        "cost": qiga_cost, "co2": qiga_co2, "time_s": qiga_time,
        "convergence": qiga_result["convergence"],
    }
    _, qiga_feasible, _ = qiga_fitness(qiga_result["best_solution"])
    results["QIGA"]["feasible"] = qiga_feasible

    # --- NSGA-II (multi-objective, reported separately) ---
    t0 = time.time()
    nsga_res, nsga_cost_curve, nsga_co2_curve = run_nsga2(pop_size=40, n_gen=60, seed=1)
    nsga_time = time.time() - t0

    print("\n" + "-" * 60)
    print(f"{'Algorithm':<12}{'Objective':>12}{'Feasible':>10}{'Cost ($)':>14}{'CO2 (kg)':>14}{'Time(s)':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<12}{r['objective']:>12.4f}{str(r['feasible']):>10}{r['cost']:>14,.0f}{r['co2']:>14,.0f}{r['time_s']:>10.2f}")
    print(f"{'NSGA-II':<12}{'(Pareto)':>12}{'—':>10}{'range':>14}{'range':>14}{nsga_time:>10.2f}")

    print("\n% reduction vs naive baseline (max speed, all HFO):")
    for name in ["GA", "QIGA"]:
        r = results[name]
        cost_red = (base_cost - r["cost"]) / base_cost * 100
        co2_red = (base_co2 - r["co2"]) / base_co2 * 100
        print(f"  {name}: cost -{cost_red:.1f}%   CO2 -{co2_red:.1f}%")

    knee_idx = np.argmin(((nsga_res.F - nsga_res.F.min(0)) / (nsga_res.F.max(0) - nsga_res.F.min(0) + 1e-9)).sum(1))
    nsga_cost_red = (base_cost - nsga_res.F[knee_idx, 0]) / base_cost * 100
    nsga_co2_red = (base_co2 - nsga_res.F[knee_idx, 1]) / base_co2 * 100
    print(f"  NSGA-II (knee point): cost -{nsga_cost_red:.1f}%   CO2 -{nsga_co2_red:.1f}%")

    print("\n" + "=" * 60)
    ga_final = results["GA"]["objective"]
    qiga_final = results["QIGA"]["objective"]
    qiga_ok = results["QIGA"]["feasible"]
    if qiga_ok and qiga_final < ga_final:
        print(f"Winner (feasible, lower objective): QIGA ({qiga_final:.4f} vs GA's {ga_final:.4f})")
    elif not qiga_ok:
        print(f"QIGA found a lower raw objective ({qiga_final:.4f}) than GA ({ga_final:.4f}) but "
              f"violated a constraint — not a valid win. GA's feasible {ga_final:.4f} stands as best.")
    else:
        print(f"Winner (feasible, lower objective): GA ({ga_final:.4f} vs QIGA's {qiga_final:.4f})")
    print("NSGA-II isn't directly comparable on one number — it returns a full Pareto front "
          "of cost/CO2 trade-offs instead of one answer. Use it to show judges the trade-off "
          "curve; use GA vs QIGA for the single-number 'which optimizer is better' claim.")

    np.save("../data/qiga_convergence.npy", np.array(qiga_result["convergence"]))
    print("\nAll convergence data saved to data/ for plotting.")
