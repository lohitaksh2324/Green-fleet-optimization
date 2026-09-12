"""
QIGA — Quantum-Inspired Genetic Algorithm (Deliverable 3).

Based on Han & Kim's Quantum-inspired Evolutionary Algorithm (QEA):
- Each gene is a qubit pair [alpha, beta], alpha^2 + beta^2 = 1 (classical encoding
  INSPIRED BY superposition, not real quantum hardware).
- "Measurement" collapses each qubit to a classical bit with probability beta^2.
- The rotation gate nudges [alpha, beta] toward the best-found solution's bits,
  replacing classical mutation — this is the core mechanism that (in theory) lets
  QIGA explore more efficiently than a classical GA's random mutation.

Encoding used here (to stay directly comparable to classical_ga_baseline.py):
  Each vessel gets SPEED_BITS bits (binary-encoded, linearly mapped to its speed
  range) and FUEL_BITS bits (binary-encoded, mod'd into a valid fuel index).
"""

import numpy as np

SPEED_BITS = 6   # 64 discrete speed levels
FUEL_BITS = 3    # 8 raw values -> mod into 6 fuel types


def bits_to_int(bits):
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def decode_individual(bits, fleet, ship_types, fuel_types):
    """Decode a flat bit array into per-vessel (speed, fuel_type)."""
    decoded = []
    idx = 0
    fuel_list = list(fuel_types.keys())
    for vessel in fleet:
        speed_bits = bits[idx: idx + SPEED_BITS]
        idx += SPEED_BITS
        fuel_bits = bits[idx: idx + FUEL_BITS]
        idx += FUEL_BITS

        lo, hi = ship_types[vessel["ship_type"]]["speed"]
        speed_int = bits_to_int(speed_bits)
        speed = lo + (speed_int / (2 ** SPEED_BITS - 1)) * (hi - lo)

        fuel_int = bits_to_int(fuel_bits) % len(fuel_list)
        fuel_type = fuel_list[fuel_int]

        decoded.append((speed, fuel_type))
    return decoded


class QIGA:
    def __init__(self, fleet, ship_types, fuel_types, fitness_fn,
                 pop_size=30, n_gen=60, rotation_angle=0.05 * np.pi,
                 catastrophe_every=20, catastrophe_frac=0.2, seed=0, repair_fn=None):
        self.fleet = fleet
        self.ship_types = ship_types
        self.fuel_types = fuel_types
        self.fitness_fn = fitness_fn  # callable(decoded_solution) -> (penalized_fitness, feasible, info)
        self.repair_fn = repair_fn    # optional callable(decoded_solution) -> repaired decoded_solution
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.rotation_angle = rotation_angle
        self.catastrophe_every = catastrophe_every
        self.catastrophe_frac = catastrophe_frac
        self.rng = np.random.default_rng(seed)

        self.n_bits = len(fleet) * (SPEED_BITS + FUEL_BITS)

    def _init_qpop(self):
        # alpha = beta = 1/sqrt(2): equal superposition, no prior bias
        return np.full((self.pop_size, self.n_bits, 2), 1 / np.sqrt(2))

    def _observe(self, qpop):
        probs = qpop[:, :, 1] ** 2  # P(bit=1) = beta^2
        rand = self.rng.random((self.pop_size, self.n_bits))
        return (rand < probs).astype(int)

    def _rotate(self, qpop, bits, best_bits, worse_mask):
        """Rotation-gate update: for individuals worse than the best, rotate each
        qubit toward the best individual's corresponding bit."""
        alpha = qpop[:, :, 0]
        beta = qpop[:, :, 1]

        # direction: +angle rotates toward bit=1, -angle rotates toward bit=0
        direction = np.where(best_bits[None, :] == 1, 1.0, -1.0)
        # only rotate where this individual's bit differs from best AND it's worse
        needs_rotation = (bits != best_bits[None, :]) & worse_mask[:, None]
        theta = np.where(needs_rotation, direction * self.rotation_angle, 0.0)

        new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
        new_beta = alpha * np.sin(theta) + beta * np.cos(theta)
        qpop[:, :, 0] = new_alpha
        qpop[:, :, 1] = new_beta
        return qpop

    def run(self):
        qpop = self._init_qpop()
        best_fitness = np.inf
        best_bits = None
        best_solution = None
        convergence = []

        for gen in range(self.n_gen):
            bits_pop = self._observe(qpop)

            fitnesses = np.zeros(self.pop_size)
            feasibles = np.zeros(self.pop_size, dtype=bool)
            for i in range(self.pop_size):
                decoded = decode_individual(bits_pop[i], self.fleet, self.ship_types, self.fuel_types)
                if self.repair_fn is not None:
                    decoded = self.repair_fn(decoded)
                fit, feasible, _ = self.fitness_fn(decoded)
                fitnesses[i] = fit
                feasibles[i] = feasible

            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_bits = bits_pop[gen_best_idx].copy()
                best_solution = decode_individual(best_bits, self.fleet, self.ship_types, self.fuel_types)
                if self.repair_fn is not None:
                    best_solution = self.repair_fn(best_solution)

            convergence.append(best_fitness)

            worse_mask = fitnesses > best_fitness
            qpop = self._rotate(qpop, bits_pop, best_bits, worse_mask)

            # catastrophe operator: periodically reinit worst fraction to escape local optima
            if self.catastrophe_every and (gen + 1) % self.catastrophe_every == 0:
                n_reset = int(self.pop_size * self.catastrophe_frac)
                worst_idx = np.argsort(fitnesses)[-n_reset:]
                qpop[worst_idx] = 1 / np.sqrt(2)

        return {
            "best_fitness": best_fitness,
            "best_bits": best_bits,
            "best_solution": best_solution,
            "convergence": convergence,
        }
