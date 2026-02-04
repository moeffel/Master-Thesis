import unittest
import numpy as np

from arima_garch.risk import value_at_risk, expected_shortfall


class TestRisk(unittest.TestCase):
    def test_value_at_risk(self):
        returns = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        alpha = 0.2
        expected = np.percentile(returns, 100 * alpha)
        self.assertAlmostEqual(value_at_risk(returns, alpha), expected)

    def test_expected_shortfall(self):
        returns = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        alpha = 0.2
        var = value_at_risk(returns, alpha)
        tail = returns[returns <= var]
        expected = tail.mean()
        self.assertAlmostEqual(expected_shortfall(returns, alpha), expected)


if __name__ == "__main__":
    unittest.main()
