import pytest
import pandas as pd

from src.data.loaders import CaliforniaHousingLoader

class TestCaliforniaHousingLoader:
    def test_load_returns_dataframe(self):
        """test that load() returns a pandas DataFrame"""
        loader = CaliforniaHousingLoader()
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
    
    def test_dataframe_not_empty(self):
        """test that DataFrame has data"""
        loader = CaliforniaHousingLoader()
        df = loader.load()
        assert len(df) > 0
        assert len(df.columns) > 0
