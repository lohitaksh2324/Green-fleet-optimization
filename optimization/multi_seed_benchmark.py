"""
Multi-seed benchmark: GA vs QIGA over N independent seeds, reporting mean/std/
win-rate instead of a single run — fixes the "did you force QIGA to win?" gap.
A single seed proves nothing; this does.
"""

import time
import numpy as np
import sys
sys.path.insert(0, "../prediction")
from fuel_simulator import SHIP_TYPES, FUEL_TYPES

from classical_ga_baseline import FLEET, run_ga
from qiga import QIGA
from algorithm_comparison import qiga_fitness

N_SEEDS = 15


def run_multi_seed():
    ga_objectives, ga_times = [], []
    qiga_objectives, qiga_times = [], []
    ga_feasible_count, qiga_feasible_count = 0, 0

    for seed in range(N_SEEDS):
        t0 = time.time()
        ga_res, _ = run_ga(pop_size=40, n_gen=60, seed=seed)
        ga_times.append(time.time() - t0)
        ga_objectives.append(ga_res.F[0])
        ga_feasible_count += int(ga_res.CV[0] <= 0)

        t0 = time.time()
        qiga = QIGA(FLEET, SHIP_TYPES, FUEL_TYPES, qiga_fitness, pop_size=40, n_gen=60, seed=seed)
        qiga_result = qiga.run()
        qiga_times.append(time.time() - t0)
        qiga_objectives.append(qiga_result["best_fitness"])
        _, feasible, _ = qiga_fitness(qiga_result["best_solution"])
        qiga_feasible_count += int(feasible)

        print(f"  seed {seed:2d}: GA={ga_objectives[-1]:.4f}  QIGA={qiga_objectives[-1]:.4f}  "
              f"{'QIGA better' if qiga_objectives[-1] < ga_objectives[-1] else 'GA better'}")

    ga_objectives = np.array(ga_objectives)
    qiga_objectives = np.array(qiga_objectives)

    print("\n" + "=" * 60)
    print(f"Results over {N_SEEDS} seeds:")
    print(f"  GA:   mean={ga_objectives.mean():.4f}  std={ga_objectives.std():.4f}  "
          f"feasible={ga_feasible_count}/{N_SEEDS}  mean_time={np.mean(ga_times):.2f}s")
    print(f"  QIGA: mean={qiga_objectives.mean():.4f}  std={qiga_objectives.std():.4f}  "
          f"feasible={qiga_feasible_count}/{N_SEEDS}  mean_time={np.mean(qiga_times):.2f}s")

    qiga_wins = int((qiga_objectives < ga_objectives).sum())
    print(f"\n  QIGA beat GA in {qiga_wins}/{N_SEEDS} seeds ({qiga_wins/N_SEEDS*100:.0f}%)")

    # simple paired significance check (Wilcoxon signed-rank; no scipy dependency needed
    # for a quick sign test as a sanity fallback)
    try:
        from scipy.stats import wilcoxon, ttest_rel
        stat, p_wilcoxon = wilcoxon(ga_objectives, qiga_objectives)
        t_stat, p_ttest = ttest_rel(ga_objectives, qiga_objectives)
        print(f"  Paired t-test p-value: {p_ttest:.4f}  |  Wilcoxon p-value: {p_wilcoxon:.4f}")
        print(f"  ({'statistically significant at p<0.05' if p_ttest < 0.05 else 'NOT statistically significant at p<0.05 — need more seeds or the difference is genuinely small'})")
    except ImportError:
        print("  (scipy not available — install for a proper significance test: pip install scipy)")

    np.save("../data/multiseed_ga_objectives.npy", ga_objectives)
    np.save("../data/multiseed_qiga_objectives.npy", qiga_objectives)
    return ga_objectives, qiga_objectives


if __name__ == "__main__":
    print(f"=== Multi-seed benchmark: GA vs QIGA, {N_SEEDS} seeds ===\n")
    run_multi_seed()
