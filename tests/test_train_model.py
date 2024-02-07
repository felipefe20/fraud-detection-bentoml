import unittest

import numpy as np
import pandas as pd
from prophet import Prophet
from train_forecaster import (
    prep_store_data,
    train_forecaster,
    train_test_split_forecaster,
)


class TestForecaster(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame(
            {
                "Date": pd.date_range(start="1/1/2023", periods=100),
                "Sales": np.random.randint(100, 1000, size=100),
                "Store": [1] * 50 + [2] * 50,
                "Open": [1] * 100,
            }
        )

    def test_prep_store_data(self):
        df_store = prep_store_data(self.df, store_id=1, store_open=1)
        self.assertTrue("ds" in df_store.columns)
        self.assertTrue("y" in df_store.columns)
        self.assertTrue(df_store["Store"].unique()[0] == 1)
        self.assertTrue(df_store["Open"].unique()[0] == 1)

    def test_train_test_split_forecaster(self):
        df_train, df_test = train_test_split_forecaster(self.df, train_fraction=0.75)
        self.assertEqual(len(df_train), 75)
        self.assertEqual(len(df_test), 25)

    def test_train_forecaster(self):
        df_train = self.df.rename(columns={"Date": "ds", "Sales": "y"})
        seasonality = {"yearly": True, "weekly": True, "daily": False}
        forecaster = train_forecaster(df_train, seasonality)
        self.assertIsInstance(forecaster, Prophet)


if __name__ == "__main__":
    unittest.main()
