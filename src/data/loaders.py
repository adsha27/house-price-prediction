import pandas as pd
import sklearn.datasets 
import logging

class CaliforniaHousingLoader:
    def load(self)->pd.DataFrame:
        logging.info("loding california housing data")
        house_price_dataset=sklearn.datasets.fetch_california_housing()
        house_price_dataframe=pd.DataFrame(
            house_price_dataset.data,
            columns=house_price_dataset.feature_names
            )
        house_price_dataframe['price']=house_price_dataset.target

        logging.info(f"Loaded{len(house_price_dataframe)} rows")
        return house_price_dataframe

        