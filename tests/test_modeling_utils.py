import importlib.util
import unittest
import numpy as np


class TestModelingUtils(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("statsmodels") and importlib.util.find_spec("arch"),
                         "statsmodels/arch not available")
    def test_invert_price_forecast(self):
        from arima_garch.modeling import invert_price_forecast

        last_price = 100.0
        log_returns = np.array([0.01, 0.0, -0.01])
        prices = invert_price_forecast(last_price, log_returns, diff_order=0)

        self.assertEqual(len(prices), 3)
        self.assertTrue(all(p is not None for p in prices))
        # First step should be > last price for positive return
        self.assertGreater(prices[0], last_price)


if __name__ == "__main__":
    unittest.main()
