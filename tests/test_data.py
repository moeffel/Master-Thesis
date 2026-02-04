import unittest
import pandas as pd
import numpy as np

from arima_garch.data import preprocess_data, train_val_test_split


class TestDataPipeline(unittest.TestCase):
    def test_preprocess_data(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "date": dates.tolist() + [dates[2]],
            "price": [100, 101, 102, 103, 104, 102],
        })
        out = preprocess_data(df)
        self.assertIn("log_return", out.columns)
        self.assertFalse(out["log_return"].isna().any())
        self.assertTrue((out["price"] > 0).all())

    def test_train_val_test_split(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "price": np.linspace(100, 200, 100),
            "log_return": np.random.normal(0, 0.01, 100),
        })
        train, val, test = train_val_test_split(df, ratios=(0.7, 0.15, 0.15), min_test_size=10)
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        self.assertGreaterEqual(len(test), 10)


if __name__ == "__main__":
    unittest.main()
