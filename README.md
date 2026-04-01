# Strandbeest Inverse Design

Conditional generative pipeline for inverse design of Strandbeest-style planar linkages.

Given a **target gait-property vector** (step length, foot clearance, duty factor, smoothness), a conditional VAE (cVAE) proposes feasible 11-parameter linkage configurations. Candidates are evaluated with an analytical forward kinematics solver and optionally post-selected by minimum target error.

---

## Linkage Parameterization

The design space follows the canonical Jansen 11-bar linkage. Each configuration is fully described by 11 link lengths $(a$–$l$, excluding $l)$, with a fixed crank pivot and a fixed support ground link.

![Canonical 11-element Jansen reference linkage](reference_schematic.png)

---

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Reproduce all results
python scripts/run_full_pipeline.py --output runs/default
```

Outputs (model weights, figures, tables) are written to `runs/default/`.

---

## Project Structure

```
src/strandbeest/
  reference.py     – canonical 11-parameter Jansen linkage definition
  kinematics.py    – forward kinematics and gait metric computation
  data.py          – dataset generation and query sampling
  models.py        – cVAE and conditional MLP (training + inference)
  baselines.py     – random search and evolutionary search
  evaluation.py    – per-query metrics (Success@ε, error, violation rate)
  pipeline.py      – end-to-end experiment orchestration
scripts/
  run_full_pipeline.py         – main reproducible entrypoint
  generate_reference_assets.py – regenerate reference schematic and CSV
tests/                         – smoke tests for kinematics solver
```

---

## Methods

| Method | Description |
|--------|-------------|
| **cVAE + post-select** | Sample K=128 latent candidates, evaluate all, return best |
| **cVAE one-shot** | Single latent sample, no evaluator calls |
| **Conditional MLP** | Deterministic regressor, perturbed at inference |
| **Evolutionary search** | Population-based, capped at B=128 evaluator calls |
| **Random search** | Uniform sampling baseline, B=128 candidates |

---

## Qualitative Results

Foot-tip trajectories produced by each method across three representative target queries. Each column corresponds to a method; each row is a different query. The cVAE + post-select approach (column 4) consistently yields smooth, closed-loop trajectories that match the target gait profile.

![Foot-trajectory comparison panel across methods and queries](trajectory_panel.png)
