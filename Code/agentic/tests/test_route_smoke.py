import tempfile
import unittest
from pathlib import Path

from agentic.scripts.route_smoke import replay


class RouteSmokeTests(unittest.TestCase):
    def test_five_route_fixtures(self):
        root = Path(__file__).resolve().parents[2]
        result = replay(root / "data" / "evaluation" / "route_manifest.jsonl")
        self.assertTrue(result["passed"])
        self.assertEqual(result["fixture_count"], 5)


if __name__ == "__main__":
    unittest.main()
