import unittest
import pandas as pd

from arima_garch.stats import difference_series


class TestStats(unittest.TestCase):
    def test_difference_series_order_1(self):
        s = pd.Series([1.0, 2.0, 4.0, 7.0])
        diff = difference_series(s, order=1)
        expected = pd.Series([1.0, 2.0, 3.0], index=s.index[1:])
        pd.testing.assert_series_equal(diff, expected)


if __name__ == "__main__":
    unittest.main()
