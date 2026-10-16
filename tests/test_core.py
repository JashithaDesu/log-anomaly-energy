"""Tests for the pieces the conclusions depend on (synthetic data, runs in seconds)."""
import unittest

import numpy as np

from logad.data import count_vectors, unique_with_counts
from logad.detectors import MLPDetector, NGramDetector, PCADetector, UnseenEvent
from logad.energy import measure
from logad.parser import Drain, grouping_accuracy

NORMAL = [(5, 5, 22, 11, 9, 26, 23, 21)] * 40 + [(5, 22, 5, 11, 9, 26, 26, 23, 21)] * 40


class TestParser(unittest.TestCase):
    def test_variables_are_masked_into_one_template(self):
        d = Drain()
        a = d.add("Received block blk_123 of size 67108864 from /10.0.0.1")
        b = d.add("Received block blk_-998 of size 512 from /10.2.3.4")
        self.assertEqual(a, b)
        self.assertEqual(d.template(a), "Received block <*> of size <*> from /<*>")

    def test_different_messages_get_different_templates(self):
        d = Drain()
        self.assertNotEqual(d.add("Deleting block blk_1 file /x"), d.add("Verification succeeded for blk_1"))

    def test_grouping_accuracy(self):
        self.assertEqual(grouping_accuracy([0, 0, 1], ["a", "a", "b"]), 1.0)
        self.assertAlmostEqual(grouping_accuracy([0, 0, 0], ["a", "a", "b"]), 0.0)


class TestData(unittest.TestCase):
    def test_unique_with_counts(self):
        u, c = unique_with_counts([(1, 2), (1, 2), (3,)])
        self.assertEqual(dict(zip(u, c)), {(1, 2): 2, (3,): 1})

    def test_count_vectors(self):
        X = count_vectors([(1, 1, 28)])
        self.assertEqual(X.shape, (1, 28))
        self.assertEqual((X[0, 0], X[0, 27]), (2, 1))


class TestDetectors(unittest.TestCase):
    def test_unseen_event_rule(self):
        d = UnseenEvent().fit(NORMAL)
        self.assertEqual(list(d.predict([NORMAL[0], (5, 99)])), [0, 1])

    def test_end_marker_catches_truncated_sessions(self):
        d = NGramDetector().fit(NORMAL).calibrate(NORMAL, 0.0)
        self.assertEqual(d.predict([NORMAL[0]])[0], 0)
        self.assertEqual(d.predict([NORMAL[0][:4]])[0], 1)      # session stops early

    def test_calibration_respects_false_positive_target(self):
        for det in (PCADetector(), NGramDetector()):
            with self.subTest(det=det.name):
                det.fit(NORMAL).calibrate(NORMAL, 0.05)
                self.assertLessEqual(det.predict(NORMAL).mean(), 0.05)

    def test_cached_and_uncached_predictions_agree(self):
        d = MLPDetector((8,), "mlp_test").fit(NORMAL * 3).calibrate(NORMAL, 0.05)
        test = NORMAL[:5] + [NORMAL[0][:3], (5, 9, 9, 9)]
        cached = d.predict(test)
        d.use_cache, d._cache = False, {}
        self.assertEqual(list(d.predict(test)), list(cached))


class TestEnergy(unittest.TestCase):
    def test_meter_reports_positive_cpu_and_energy(self):
        with measure() as m:
            sum(i * i for i in range(200000))
        self.assertGreater(m.cpu_s, 0)
        self.assertGreater(m.energy_j, 0)
        self.assertIn(m.energy_source.split()[0], ("measured", "estimated"))


if __name__ == "__main__":
    unittest.main()
