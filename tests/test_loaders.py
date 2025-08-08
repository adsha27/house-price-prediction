import pytest
import pandas as pd
from unittest.mock import patch
from src.data.loaders import CaliforniaHousingLoader, AmesHousingLoader

class TestCaliforniaHousingLoader:
    def test_load_returns_dataframe(self):
        """test that load() returns a pandas DataFrame"""
        loader = CaliforniaHousingLoader()
        result = loader.load()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestAmesHousingLoader:
    def test_load_returns_dataframe(self, tmp_path):
        #dummy for test
        d = tmp_path / "data/raw"
        d.mkdir(parents=True)
        p = d / "AmesHousing.csv"
        p.write_text("Id,SalePrice\n1,200000")

        with patch('src.data.loaders.AmesHousingLoader.DATA_PATH', p):
            loader = AmesHousingLoader()
            result = loader.load()
            assert isinstance(result, pd.DataFrame)
            assert not result.empty

    def test_renames_saleprice_column(self, tmp_path):
        """ testing renaming of the SalePrice Column"""
        d = tmp_path / "data/raw"
        d.mkdir(parents=True)
        p = d / "AmesHousing.csv"
        p.write_text("Id,SalePrice\n1,200000")

        with patch('src.data.loaders.AmesHousingLoader.DATA_PATH', p):
            loader = AmesHousingLoader()
            df = loader.load()
            assert 'price' in df.columns
            assert 'SalePrice' not in df.columns