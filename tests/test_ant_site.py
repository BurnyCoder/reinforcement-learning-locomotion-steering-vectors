"""Check publication-boundary failures without importing the experiment stack.

Standard-library isolation: https://docs.python.org/3/library/unittest.html
and https://docs.python.org/3/library/tempfile.html.
"""
import tempfile
import unittest
from pathlib import Path

from scripts.check_ant_site import check_arm, load_json, local_file


class AntSiteTests(unittest.TestCase):
    """Reject missing, escaping and malformed public artifacts before deployment."""

    def test_local_asset_must_exist_inside_site(self):
        """Resolve a real asset while rejecting escape paths and missing files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset.json").write_text("{}", encoding="utf-8")
            self.assertEqual(local_file(root, "asset.json"), root / "asset.json")
            for reference in ("../asset.json", "%2e%2e/asset.json", "/asset.json", "C:/asset.json", "missing.json", "data\\asset.json"):
                with self.subTest(reference=reference), self.assertRaises(ValueError):
                    local_file(root, reference)

    def test_nonfinite_json_is_rejected(self):
        """Reject explicit and overflow-produced nonfinite values, including nested ones."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            for literal in ("NaN", "Infinity", "-Infinity", "1e309"):
                path.write_text('{"nested": [' + literal + "]}", encoding="utf-8")
                with self.subTest(literal=literal), self.assertRaises(ValueError):
                    load_json(path)

    def test_episode_preserves_the_recorded_endpoint(self):
        """An early failed episode remains valid; an omitted endpoint is not."""
        arm = {"length": 3, "failed": True, "prefix_only": True, "steps": [0, 3],
               "time": [0.0, 0.15], "x": [0.0, 0.1], "y": [0.0, 0.2], "reward": [None, 1.0]}
        check_arm(arm, warmup=100, context="fixture")
        arm["steps"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "endpoint"):
            check_arm(arm, warmup=100, context="fixture")

    def test_inconsistent_curve_lengths_are_rejected(self):
        """Prevent shifted or truncated chart arrays from silently reaching the browser."""
        arm = {"length": 3, "failed": False, "prefix_only": False, "steps": [0, 3],
               "time": [0.0, 0.15], "x": [0.0], "y": [0.0, 0.2], "reward": [None, 1.0]}
        with self.assertRaisesRegex(ValueError, "length"):
            check_arm(arm, warmup=100, context="fixture")


if __name__ == "__main__":
    unittest.main()
