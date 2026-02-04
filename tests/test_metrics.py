import math
import unittest

import numpy as np

from arima_garch.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    qlike_loss_calc,
    qlike_loss,
)


class TestMetrics(unittest.TestCase):
    def test_basic_errors(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 1.0 / 3.0)
        self.assertAlmostEqual(mean_squared_error(y_true, y_pred), 1.0 / 3.0)
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), math.sqrt(1.0 / 3.0))

    def test_mape(self):
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([110.0, 190.0, 420.0])
        mape = mean_absolute_percentage_error(y_true, y_pred)
        self.assertGreater(mape, 0)
        self.assertAlmostEqual(mape, 6.666666666666667, places=6)

    def test_qlike(self):
        actual = np.array([1.0, 2.0, 4.0])
        forecast = np.array([1.0, 2.0, 4.0])
        # For perfect forecasts, qlike should be 0
        self.assertAlmostEqual(qlike_loss(actual, forecast), 0.0)
        self.assertAlmostEqual(qlike_loss_calc(2.0, 2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
