"""Local: test signed strength parsing; global: reject ambiguous prespecified searches.

Sources: https://docs.python.org/3.12/library/math.html#math.isfinite
https://docs.python.org/3.12/library/string.html#format-specification-mini-language
"""
import pytest  # Invalid protocols must fail before simulation or cached artifact access.

from rl_locomotion_steering_vectors.config import Config, read_config  # Exercise direct manifests and .env entry points.
from rl_locomotion_steering_vectors.experiment import condition_name  # Check actual cache labels rather than a duplicate formatter.


def test_default_and_fine_strength_grids():
    """Local: parse numbers without changing order; global: preserve the original default search."""
    assert Config().strengths == (-.5, -.2, -.1, -.05, .05, .1, .2, .5)  # Existing manifests lacking the new field retain their coarse grid.
    config = Config(strength_grid=" -0.025, -5e-3, +0.005, 0.025 ")  # Whitespace and scientific notation are normal float inputs.
    assert config.strengths == (-.025, -.005, .005, .025)  # Preserve the declared validation order and signed values.
    assert len({condition_name("speed", value) for value in config.strengths}) == 4  # Distinct grid entries must map to distinct cache directories.
    assert Config(**config.as_dict()).strengths == config.strengths  # Manifest round trips retain the complete search specification.


@pytest.mark.parametrize("grid", ["", "-0.1,", "-0.1,nope", "-0.1,nan", "-inf,0.1", "-0.1,0,0.1", "-0.1,0.1,0.10", "0.1,0.2", "-0.1,-0.2", "-0.1,0.01001,0.01002", "-0.1,0.00001"])
def test_invalid_strength_grid_is_rejected_before_run(grid):
    """Local: cover malformed, unsafe, and colliding values; global: prevent silent search changes."""
    with pytest.raises(ValueError, match="strength"):
        Config(strength_grid=grid)  # Direct construction also covers configuration restored from saved manifests.


def test_env_strength_grid_keeps_typed_numeric_settings(tmp_path, monkeypatch):
    """Local: load the new string alongside scalar settings; global: support the documented user command."""
    monkeypatch.chdir(tmp_path)  # An isolated local .env prevents the real research protocol from changing.
    (tmp_path / ".env").write_text("STEERING_STRENGTH_GRID=-0.02,0.02\nSTEERING_MAX_STEPS=400\n", encoding="utf-8")  # Use the actual dotenv parser.
    for key in Config().as_dict():
        monkeypatch.delenv(f"STEERING_{key.upper()}", raising=False)  # Ambient experiment settings cannot alter the fixture.
    config = read_config()  # The user-facing configuration path must accept strings without numeric comparisons.
    assert config.strengths == (-.02, .02) and config.max_steps == 400  # Parsing must preserve both data types.
