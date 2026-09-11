"""
FastAPI backend — Deliverable 4 (Software Platform / Decision Support System).

Endpoints:
  GET  /health                  - liveness check
  POST /predict                 - fuel consumption for one voyage (XGBoost model)
  POST /optimize                - run QIGA on a fleet spec, return best solution
  POST /optimize/scenario       - run QIGA under a named fuel-adoption scenario
  GET  /scenarios                - list available fuel-adoption scenarios
  POST /report/generate          - build the PDF technical report, returns the file

Design choice: optimization runs are synchronous for small fleets (<=200 vessels,
finishes in seconds) but the endpoint accepts a `pop_size`/`n_gen` override so the
dashboard can request a fast "preview" run vs a slower "final" run — this is the
practical fix for the interactivity-vs-runtime tradeoff flagged during scale testing
(1440 vessels takes ~170s, too slow for a click-and-wait UI at full budget).
"""

import sys
import time
sys.path.insert(0, "../prediction")
sys.path.insert(0, "../optimization")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np

from fuel_simulator import SHIP_TYPES, FUEL_TYPES
from classical_ga_baseline import (
    voyage_fuel_and_cost, W_COST, W_CO2, FUEL_ADOPTION_CAP,
    CAPPED_FUELS, max_allowed_count, CII_REDUCTION_TARGET, vessel_naive_co2_per_nm,
)
from qiga import QIGA
from scale_test import generate_fleet, naive_baseline, repair_decoded_solution
import report_generator

app = FastAPI(title="Egreen Quanta API", version="0.1.0")

MODEL_PATH = "../data/xgb_fuel_model.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


# ---------- Scenarios (Day 10 of the plan) ----------
SCENARIOS = {
    "diesel_only": {"HFO": 1.0, "MDO": 1.0, "LNG": 0.0, "Methanol": 0.0, "Hydrogen": 0.0, "Ammonia": 0.0},
    "lng_mix_30pct": {"HFO": 1.0, "MDO": 1.0, "LNG": 0.30, "Methanol": 0.0, "Hydrogen": 0.0, "Ammonia": 0.0},
    "current_baseline": FUEL_ADOPTION_CAP,  # the caps already calibrated (LNG 50%, Methanol 30%, H2 15%, NH3 10%)
    "aggressive_hydrogen": {"HFO": 1.0, "MDO": 1.0, "LNG": 0.50, "Methanol": 0.30, "Hydrogen": 0.40, "Ammonia": 0.20},
}


# ---------- Request/response models ----------
class VoyageRequest(BaseModel):
    ship_type: str = Field(..., description=f"One of {list(SHIP_TYPES.keys())}")
    fuel_type: str = Field(..., description=f"One of {list(FUEL_TYPES.keys())}")
    weather: str = Field("Moderate", description="Calm, Moderate, or Stormy")
    displacement_tons: float
    cargo_load_pct: float = Field(0.8, ge=0.0, le=1.0)
    speed_knots: float
    distance_nm: float


class PredictionResponse(BaseModel):
    predicted_fuel_kg: float
    ship_type: str
    fuel_type: str


class OptimizeRequest(BaseModel):
    n_vessels: int = Field(8, ge=1, le=1440, description="Fleet size, up to the tested 1440 max")
    pop_size: int = Field(40, description="Lower for a fast preview, higher for a final run")
    n_gen: int = Field(60, description="Lower for a fast preview, higher for a final run")
    scenario: Optional[str] = Field(None, description=f"One of {list(SCENARIOS.keys())}, or omit for default caps")
    seed: int = 1


class VesselAllocation(BaseModel):
    ship_type: str
    speed_knots: float
    fuel_type: str


class OptimizeResponse(BaseModel):
    objective: float
    feasible: bool
    total_cost_usd: float
    total_co2_kg: float
    runtime_seconds: float
    allocation: List[VesselAllocation]
    scenario_used: Optional[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: VoyageRequest):
    if req.ship_type not in SHIP_TYPES:
        raise HTTPException(400, f"Unknown ship_type. Must be one of {list(SHIP_TYPES.keys())}")
    if req.fuel_type not in FUEL_TYPES:
        raise HTTPException(400, f"Unknown fuel_type. Must be one of {list(FUEL_TYPES.keys())}")

    model = get_model()
    voyage_hours = req.distance_nm / req.speed_knots
    row = pd.DataFrame([{
        "displacement_tons": req.displacement_tons,
        "cargo_load_pct": req.cargo_load_pct,
        "speed_knots": req.speed_knots,
        "distance_nm": req.distance_nm,
        "voyage_hours": voyage_hours,
        "ship_type": req.ship_type,
        "fuel_type": req.fuel_type,
        "weather": req.weather,
    }])
    log_pred = model.predict(row)[0]
    fuel_kg = float(np.expm1(log_pred))
    return PredictionResponse(predicted_fuel_kg=fuel_kg, ship_type=req.ship_type, fuel_type=req.fuel_type)


@app.get("/scenarios")
def list_scenarios():
    return SCENARIOS


def _run_optimization(n_vessels, pop_size, n_gen, seed, fuel_caps):
    fleet = generate_fleet(n_vessels, seed=7)
    base_cost, base_co2 = naive_baseline(fleet)
    max_hours = (900 / 8) * n_vessels

    capped_fuels_local = [f for f, cap in fuel_caps.items() if cap < 1.0]

    def max_allowed_local(fuel_type, n):
        import math
        return math.ceil(fuel_caps[fuel_type] * n)

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
        g_fuel = [fuel_counts[f] - max_allowed_local(f, len(fleet)) for f in capped_fuels_local]
        feasible = (g1 <= 1e-6) and (cii_violation <= 1e-6) and all(g <= 0 for g in g_fuel)
        penalty = max(0, g1) * 0.01 + cii_violation * 0.5 + sum(max(0, g) for g in g_fuel) * 0.3
        return objective + penalty, feasible, {"total_cost": total_cost, "total_co2": total_co2}

    def repair(decoded):
        # scenario-aware repair (mirrors scale_test.repair_decoded_solution but uses
        # this request's fuel_caps instead of the module-level default caps)
        speeds = [s for s, f in decoded]
        fuels = [f for s, f in decoded]
        counts = {f: 0 for f in capped_fuels_local}
        cleanest_first = sorted(capped_fuels_local, key=lambda f: FUEL_TYPES[f]["co2_per_kg"])
        for i, f in enumerate(fuels):
            if f not in capped_fuels_local:
                continue
            cap = max_allowed_local(f, len(fleet))
            if counts[f] < cap:
                counts[f] += 1
                continue
            placed = False
            for alt in cleanest_first:
                if counts[alt] < max_allowed_local(alt, len(fleet)):
                    fuels[i] = alt
                    counts[alt] += 1
                    placed = True
                    break
            if not placed:
                fuels[i] = "MDO"
        repaired = list(zip(speeds, fuels))
        # CII speed repair
        final = []
        for (speed, fuel), vessel in zip(repaired, fleet):
            lo, hi = SHIP_TYPES[vessel["ship_type"]]["speed"]
            fuel_kg, co2_kg, cost_usd, hours = voyage_fuel_and_cost(vessel, speed, fuel)
            actual = co2_kg / vessel["distance_nm"]
            target = CII_REDUCTION_TARGET * vessel_naive_co2_per_nm(vessel)
            if actual > target and actual > 0:
                speed = max(lo, min(hi, speed * np.sqrt(target / actual)))
            final.append((speed, fuel))
        return final

    t0 = time.time()
    qiga = QIGA(fleet, SHIP_TYPES, FUEL_TYPES, fitness, pop_size=pop_size, n_gen=n_gen,
                seed=seed, repair_fn=repair)
    result = qiga.run()
    runtime = time.time() - t0

    _, feasible, info = fitness(result["best_solution"])
    allocation = [
        {"ship_type": v["ship_type"], "speed_knots": round(s, 2), "fuel_type": f}
        for (s, f), v in zip(result["best_solution"], fleet)
    ]
    return {
        "objective": result["best_fitness"],
        "feasible": feasible,
        "total_cost_usd": info["total_cost"],
        "total_co2_kg": info["total_co2"],
        "runtime_seconds": runtime,
        "allocation": allocation,
    }


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    fuel_caps = FUEL_ADOPTION_CAP
    result = _run_optimization(req.n_vessels, req.pop_size, req.n_gen, req.seed, fuel_caps)
    return OptimizeResponse(**result, scenario_used=None)


@app.post("/optimize/scenario/{scenario_name}", response_model=OptimizeResponse)
def optimize_scenario(scenario_name: str, req: OptimizeRequest):
    if scenario_name not in SCENARIOS:
        raise HTTPException(400, f"Unknown scenario. Must be one of {list(SCENARIOS.keys())}")
    fuel_caps = SCENARIOS[scenario_name]
    result = _run_optimization(req.n_vessels, req.pop_size, req.n_gen, req.seed, fuel_caps)
    return OptimizeResponse(**result, scenario_used=scenario_name)


@app.post("/report/generate")
def generate_report():
    """Builds the technical report PDF from the saved experiment data (charts,
    tables, scenario comparison — see report_generator.py) and returns the file
    for download. Takes ~15-20s (runs a live QIGA pass for the scenario chart)."""
    try:
        path = report_generator.build_report()
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename="Egreen_Quanta_Technical_Report.pdf",
    )
