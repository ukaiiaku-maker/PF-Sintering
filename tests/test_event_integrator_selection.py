import unittest

from pf_sintering.event_integrator_selection import select_event_integrator


class EventIntegratorSelectionTests(unittest.TestCase):
    def test_auto_timescale_selection(self):
        cases = (
            (0.01, "packet", False),
            (100.0, "quasistatic", False),
            (1.0, "packet", True),
        )
        for ratio, selected, warning in cases:
            with self.subTest(ratio=ratio):
                result = select_event_integrator(
                    "auto", tau_GB=ratio, tau_surface=1.0)
                self.assertEqual(result["event_integrator_selected"], selected)
                self.assertIs(
                    result["intermediate_timescale_warning"], warning)

    def test_manual_overrides_bypass_auto_selection(self):
        for override in ("packet", "quasistatic"):
            with self.subTest(override=override):
                # Choose a ratio that would select the opposite auto method.
                ratio = 100.0 if override == "packet" else 0.01
                result = select_event_integrator(
                    override, tau_GB=ratio, tau_surface=1.0)
                self.assertEqual(
                    result["event_integrator_selected"], override)
                self.assertEqual(result["selection_reason"], "manual override")
                self.assertIs(
                    result["intermediate_timescale_warning"], False)


if __name__ == "__main__":
    unittest.main()
