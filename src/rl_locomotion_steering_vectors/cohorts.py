"""Local: check committed seed reservations. Global: distinguish old reproduction from fresh allocation.

The ledger backfills Ant 005's manifest-omitted application reservation and its
later audit consumption; it does not change either historical experiment.
Sources: reports/fact-check-2026-10-22.md and Python package-resource API
https://docs.python.org/3.12/library/importlib.resources.html#importlib.resources.files .
"""
from importlib.resources import files  # Local: locate packaged metadata; global: avoid depending on the shell's working directory.
import json  # Local: parse non-executable records; global: keep reservations inspectable.


def load_cohorts() -> list[dict]:
    """Local: read the committed ledger. Global: share the same reservations between application and audit guards."""
    record = json.loads(files(__package__).joinpath("seed_cohorts.json").read_text(encoding="utf-8"))  # Local: use a package resource; global: include the ledger in installed distributions.
    if record["schema"] != 1:
        raise ValueError("Unsupported seed-cohort registry schema")  # Local: reject an unknown interpretation; global: do not silently ignore reservations.
    return record["cohorts"]  # Local: expose the explicit records; global: allow reports to cite the same ledger.


def assert_available_cohort(model_key: str, seeds, *, resume_cohort: str | None = None) -> dict:
    """Local: reject overlap or verify an exact named resume. Global: never count previously consumed seeds as new evidence.

This is a read-only guard against the committed ledger, not an automatic allocator.
New cohorts still need a saved protocol/ledger entry before collection. Treatment,
policy and runtime identities remain enforced by the existing episode-cache guards.
"""
    requested = list(seeds)  # Local: consume iterators once; global: inspect the entire intended cohort.
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("A seed cohort must be nonempty and contain unique seeds")  # Local: reject ambiguous units; global: preserve the declared denominator.
    seed_set, rows = set(requested), load_cohorts()  # Local: use set intersection; global: compare all recorded reservations.
    overlaps = [row for row in rows if row["model_key"] == model_key and seed_set.intersection(range(row["start"], row["stop"]))]  # Local: stop is exclusive like Python range; global: include consumed and reserved cohorts alike.
    if resume_cohort is not None:
        matches = [row for row in rows if row["id"] == resume_cohort]  # Local: look up the named historical owner; global: a caller cannot invent a bypass.
        if len(matches) != 1:
            raise ValueError(f"Unknown historical seed cohort: {resume_cohort}")  # Local: require one recorded owner; global: reject typos and missing provenance.
        own = matches[0]  # Local: resolve the unique ledger row; global: compare its complete original allocation.
        if own["model_key"] != model_key or seed_set != set(range(own["start"], own["stop"])) or own["status"] != "consumed":
            raise ValueError("Historical resume requires the exact consumed cohort and model")  # Local: reject subsets, extensions and unused reservations; global: no post-hoc cohort selection.
        conflicts = [row for row in overlaps if row["id"] != resume_cohort and row.get("superseded_by") != resume_cohort]  # Local: permit only the explicitly documented superseded reservation; global: do not excuse another overlap.
    else:
        conflicts = overlaps  # Local: every existing reservation blocks a new allocation; global: fresh means outside the ledger.
    if conflicts:
        collisions = sorted(seed_set.intersection(seed for row in conflicts for seed in range(row["start"], row["stop"])))  # Local: name exact reused resets; global: make the defect actionable.
        owners = ", ".join(row["id"] for row in conflicts)  # Local: explain each reservation; global: preserve the evidence chain in errors/logs.
        raise ValueError(f"Seed cohort overlap {collisions} with {owners}")  # Local: stop before collection; global: prevent misleading fresh-evidence claims.
    return {"mode": "historical_resume" if resume_cohort else "new_allocation", "cohort": resume_cohort,
            "overlaps": [row["id"] for row in overlaps]}  # Local: expose the checked mode and known overlap; global: keep historical collision disclosure visible.
