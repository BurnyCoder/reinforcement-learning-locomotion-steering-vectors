# hc-classic-001: extraction rejected before validation

This bundle preserves the first HalfCheetah fitting experiment. Its raw vectors were rejected after fitting-only sensitivity analysis found substantial influence from a few near-stationary observations with accumulated torso rotations. No validation, confirmation, or causal steering success is claimed.

- [Measured run report](report.md) and [visually checked PDF](report.pdf).
- [Detailed diagnostic reasoning and sensitivity analysis](../hc-classic-001-diagnostics.md), kept separately from the generated report.
- [Exact manifest](manifest.json), [complete recorded results](results.json), and [vector diagnostics](vector_diagnostics.json). JSON whitespace is compacted; all source values are preserved.
- [Original rejected vectors](vectors.npz), retained for audit rather than recommended use.

The corresponding local run is `runs/hc-classic-001`. Original fitting trajectories are retained there and are not included in this compact Git bundle. There are no strength-response charts because the experiment stopped before validation. The full seed lists in the manifest are registered protocol assignments; the later evaluation splits were not used.

The PDF was regenerated from saved evidence and both pages were rendered with Poppler and visually inspected. See the [experiment register](../experiments.md) for the follow-up hypothesis.
