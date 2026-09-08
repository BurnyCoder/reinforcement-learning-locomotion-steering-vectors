# Reproducing the Ant fitting inspection

`fitting-inspection.json` is a byte-for-byte copy of the original locally recorded `artifacts/runtime-smoke/ant-classic-005-fitting-inspection.json`. Its original scope, including the statement that validation was not read at that time, is preserved as historical metadata. The later verification does not independently establish that chronology.

From the project root, with the locked environment installed and original saved `runs` restored, run:

```powershell
uv run --no-sync python scripts/audit_documentation.py
```

The command reads the existing diagnostic/fitting NPZ files and independently recalculates completion counts, early endings, physical episode means/ranges, complete-window groups, label contrasts, sign consistency, heading association, adjacent-window correlation, first-window exclusion, completion-only exclusion, one-percent label-tail trimming, every leave-one-episode-out refit, and 200/300-step window contrasts. It also checks quaternion-derived yaw rates wherever a successor observation exists. No simulator, policy inference, new rollout, or changed experimental selection is involved.

The fresh calculation and every discrepancy are saved in `reports/documentation-audit/ant-sensitivity-verification.json`. The full audit additionally writes `numerical-audit.json`, `input-identities.json`, and a timestamped complete terminal/file log. Inputs are hashed before use and checked again after calculation. A missing raw dataset or mismatched calculation makes the command exit unsuccessfully.

The independent arithmetic is in [`documentation_evidence.py`](../../../scripts/documentation_evidence.py), with phase comparisons in [`documentation_checks.py`](../../../scripts/documentation_checks.py). Quartiles use NumPy's [linear quantile convention](https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html). Behavior/activation contrasts average within episode before averaging episodes; the original descriptive heading statistic instead averages its arithmetic window headings equally. The separate later causal audit uses circular heading means, so those descriptive heading numbers are intentionally different.

The original sensitivity directions are recalculated only to verify historical diagnostics. They are never saved over the selected vector or evaluated as new interventions. Differences are checked at absolute tolerance `1e-12`; claims published only as rounded prose are compared at their stated precision where explicitly labeled.
