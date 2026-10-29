"""Local: test recorded seed reservations. Global: keep historical replay separate from fresh evidence.

Source: the documented application-cohort collision in reports/fact-check-2026-10-22.md.
These tests use seed identities only and never collect scientific episodes.
"""
import pytest  # Local: assert rejected allocations; global: prevent silent evidence reuse.

from rl_locomotion_steering_vectors.cohorts import assert_available_cohort, load_cohorts  # Local: exercise the shared ledger guard; global: use one freshness rule.
from rl_locomotion_steering_vectors.demonstration import fresh_application_seeds  # Local: cover the public application helper; global: catch the original regression.


def test_ant_application_rejects_consumed_audit_seeds():
    """Local: reproduce the omitted-manifest collision. Global: prevent another application claiming these resets are fresh."""
    manifest = {"model_key": "ant", "config": {"seed_offset": 500000}, "seed_splits": {"replication": [530000]}}  # Local: omit the audit as historical manifests do; global: require the external ledger.
    with pytest.raises(ValueError, match="540000.*ant-005-independent-audit"):
        fresh_application_seeds(manifest)  # Local: request the former application reservation; global: reject before policy loading.


@pytest.mark.parametrize("seeds", [[500000], [501001], [510002], [520003], [530004], [540019]])
def test_new_ant_allocation_rejects_every_recorded_phase(seeds):
    """Local: cover consumed and reserved ranges. Global: protect all Ant evidence partitions."""
    with pytest.raises(ValueError, match="overlap"):
        assert_available_cohort("ant", seeds)  # Local: no resume identity; global: a new allocation cannot borrow old resets.


def test_exact_historical_audit_resume_allows_its_superseded_reservation():
    """Local: acknowledge the recorded overlap. Global: permit reproduction without treating it as a new allocation."""
    assert_available_cohort("ant", range(540000, 540020), resume_cohort="ant-005-independent-audit:audit")  # Local: exact historical cohort; global: retain reproducibility.
    with pytest.raises(ValueError, match="exact"):
        assert_available_cohort("ant", range(540000, 540021), resume_cohort="ant-005-independent-audit:audit")  # Local: append one unseen seed; global: disallow silently expanding an observed cohort.
    with pytest.raises(ValueError, match="exact"):
        assert_available_cohort("ant", range(540000, 540010), resume_cohort="ant-005-independent-audit:audit")  # Local: remove half the cohort; global: disallow survivor-selected replay as a cohort resume.


def test_seed_identity_includes_environment_and_resume_owner():
    """Local: distinguish task-specific reset spaces. Global: do not falsely collide unrelated environments or guessed owners."""
    assert_available_cohort("halfcheetah", [540000])  # Local: Ant ledger does not reserve HalfCheetah resets; global: retain existing experiments.
    assert_available_cohort("ant", [900000])  # Local: unused Ant seed; global: allow a genuinely new allocation.
    with pytest.raises(ValueError, match="Unknown"):
        assert_available_cohort("ant", [900000], resume_cohort="invented:audit")  # Local: nonexistent owner; global: no blanket bypass.
    with pytest.raises(ValueError, match="exact"):
        assert_available_cohort("halfcheetah", range(540000, 540020), resume_cohort="ant-005-independent-audit:audit")  # Local: wrong environment; global: resume requires the complete identity.


def test_ledger_records_unused_replication_and_consumed_application_overlap():
    """Local: inspect the explicit backfill. Global: retain the historical reservation without calling it fresh."""
    rows = {row["id"]: row for row in load_cohorts()}  # Local: index the committed records; global: test packaged evidence metadata.
    assert rows["ant-classic-005:replication"]["status"] == "reserved"  # Local: never run; global: avoid fabricating replication.
    assert rows["ant-classic-005:application"]["superseded_by"] == "ant-005-independent-audit:audit"  # Local: disclose the prior collision; global: prevent future reuse.
    assert rows["ant-005-independent-audit:audit"]["status"] == "consumed"  # Local: completed reset cohort; global: protect fresh-evidence claims.
