from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from eeg_seizure.modeling import add_window_metadata
from eeg_seizure.visualization import merge_intervals


class DatasetArrays(dict):
    @property
    def files(self):
        return list(self.keys())


class PredictionMetadataTests(unittest.TestCase):
    def test_attaches_record_and_window_timing(self):
        dataset = DatasetArrays(
            files=np.array(["chb01_01.edf", "chb01_01.edf"]),
            window_start_sec=np.array([0.0, 2.0]),
            window_sec=np.array([4.0]),
        )
        predictions = pd.DataFrame({"y_true": [0, 1], "y_prob": [0.1, 0.8]})

        result = add_window_metadata(predictions, dataset)

        self.assertEqual(result["file"].tolist(), ["chb01_01.edf", "chb01_01.edf"])
        self.assertEqual(result["window_start_sec"].tolist(), [0.0, 2.0])
        self.assertEqual(result["window_sec"].tolist(), [4.0, 4.0])
        self.assertNotIn("file", predictions.columns)

    def test_rejects_misaligned_window_metadata(self):
        dataset = DatasetArrays(
            files=np.array(["chb01_01.edf"]),
            window_start_sec=np.array([0.0, 2.0]),
            window_sec=np.array([4.0]),
        )
        predictions = pd.DataFrame({"y_true": [0, 1], "y_prob": [0.1, 0.8]})

        with self.assertRaisesRegex(ValueError, "length"):
            add_window_metadata(predictions, dataset)

    def test_merges_overlapping_and_adjacent_intervals(self):
        self.assertEqual(
            merge_intervals([(5.0, 8.0), (0.0, 2.0), (2.0, 4.0), (7.0, 10.0)]),
            [(0.0, 4.0), (5.0, 10.0)],
        )

    def test_rejects_reversed_intervals(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            merge_intervals([(3.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
