"""The scenario simulator (code/sim/): the real ears + barge-in + conversation in a simulated room, one test per
scenario. A scenario with a known, not yet fixed gap is an expected failure; the ones replaying the real barge-in
recording are skipped where it is not on the machine (family voice data, local only)."""
import logging
import unittest

from sim.scenarios import SCENARIOS, run_one


def _make(sc):
    def test(self):
        r = run_one(sc, seed=0)
        if r.status == "SKIP":
            self.skipTest(r.reason())
        self.assertEqual(r.status if r.status != "XFAIL" else "FAIL", "PASS", r.reason())
    test.__doc__ = sc["title"]
    return unittest.expectedFailure(test) if sc["known"] else test


class ScenarioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        logging.disable(logging.NOTSET)


for _sc in SCENARIOS:
    setattr(ScenarioTest, "test_" + _sc["name"], _make(_sc))


if __name__ == "__main__":
    unittest.main()
