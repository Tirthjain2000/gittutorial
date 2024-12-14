import datetime
from collections import namedtuple
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from app.base_price_estimate import VERSION, get_model_location
from app.base_price_estimate.data_models import BasePriceCountry
from app.base_price_estimate.utils import feature_engineering
from app.base_price_model.models import LightGBMRegressionModel
from danamica_utils.dates import d_week, d_year, list_of_date


class BasePricePredictor:

    price_columns = {
        "base_price_recent": ["price", "price_q0.1", "price_q0.5", "price_q0.9"],
        "base_price": ["price", "price_upper", "price_lower"],
        "": ["price", "price_upper", "price_lower"],
    }

    def __init__(self, input_data: Dict[str, Any], model_name: str = "", version: str = None,model=None) -> None:
        """
        input_data: Dict[str, Any] consists of

            data = {
                country: str, required,
                apartment: bool, required,
                price_method: str, required,
                release_start: str, required,
                release_end: str, required,
                lat: float, required,
                lng: float, required,
                number_of_persons: int, required,
                bathroom_count: int, required,
                bedroom_count: int, required,
                has_pool: bool, optional,
                shared_pool: bool, optional,
                has_wifi: bool, optional,
                jacuzzi: bool, optional,
                sauna: bool, optional,
                balcony_or_terrace: bool, optional,
                aircondition: bool, optional,
                house_size: int, optional,
                kitchen: bool, optional,
                is_detached: bool, optional,
                in_ski_area: bool, optional,
                is_park: bool, optional,
                beach_lake_front: bool, optional,
                bbq: bool, optional,
                child_friendly: bool, optional,
                gym_fitness: bool, optional,
                parking: bool, optional,
                garden: bool, optional,
                pets_allowed: bool, optional,
            }


        Example:

            >>> base_price_predictor = BasePricePredictor(input_data)
            >>> # async_predict
            >>> data = base_price_predictor.async_predict()
            >>> data = base_price_predictor.predict()

        """
        self.input_data = input_data
        self.country_code: BasePriceCountry = input_data["country"]
        self.model_name = model_name
        self.version = version
        self.model = model

    async def async_predict(self) -> Dict[str, Any]:
        """Main method that return the prices"""
        response = await self.predict()

        return response

    def prepare_df_input(self, specific_year: Optional[int] = None) -> pd.DataFrame:
        # Crate DataFrame from data
        df: pd.DataFrame = self.data_to_df(data=self.input_data, specific_year=specific_year)

        # add same features as during training
        df = feature_engineering(df=df)

        return df

    async def predict(self) -> Dict[str, Any]:
        """Contains flow of changes applied to the raw predictions"""
        df = self.get_model_prediction()

        # Align differences between years
        # we only allow for small diff in average price per year
        df = self.align_prices_between_years(df=df)

        # Adjust for weekend prices
        # ie. ensure profile with higher night price in weekends
        df = self.predict_weekend(df_7d=df)

        # Adjust prices manully based on presence of certain features
        df = self.manual_overrides(df=df)

        # Ensure price_upper > price > price_lower
        df = self.ensure_upper_lower(model_name=self.model_name, df=df)

        df["date_in"] = df["date_in"].astype(str)

        # Check if there are outliers in the features
        outliers = self.get_outliers(df=df)

        return self.construct_response(df=df, outliers=outliers)

    def get_model_prediction(self, specific_year: Optional[int] = None) -> pd.DataFrame:
        df_input = self.prepare_df_input(specific_year)

        if not isinstance(df_input, pd.DataFrame):
            print("df_input is not a dataframe")
            return None

        # Get predictions
        df_prices = self._get_model_predictions(df=df_input)

        if self.model_name != "base_price_recent":
            # Rename columns
            df_prices = df_prices.rename(
                columns={
                    "price": "price",
                    "price_q0.1": "price_lower",
                    "price_q0.9": "price_upper",
                }
            )

        df_prices = df_prices[["date_in"] + self.price_columns[self.model_name]]

        return df_prices

    def _get_model_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [x for x in self.model.ordered_columns if x not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        return self.model.predict(df[self.model.ordered_columns], quantiles=True, transformer=np.exp)

    def predict_weekend(self, df_7d: pd.DataFrame) -> pd.DataFrame:
        """Weekday / weekend pricing"""
        self.input_data["duration"] = 2
        df_1d = self.get_model_prediction()

        new_cols = [f"{col}_1d" for col in self.price_columns[self.model_name]]
        # Rename columns names
        df_1d.rename(
            columns=dict(zip(self.price_columns[self.model_name], new_cols)),
            inplace=True,
        )

        df_1d = df_1d[["date_in"] + new_cols]

        df = df_1d.merge(df_7d, on="date_in")

        df["week"] = df["date_in"].apply(d_week)
        df["year"] = df["date_in"].apply(d_year)
        df["date_in"] = pd.to_datetime(df["date_in"])
        df["dayofweek"] = df["date_in"].dt.dayofweek

        df["week_price_1d"] = df.groupby(["week", "year"], sort=False)["price_1d"].transform("mean")
        df["week_price_7d"] = df.groupby(["week", "year"], sort=False)["price"].transform("mean")

        df["week_price_7d_sum"] = df.groupby(["week", "year"], sort=False)["price"].transform("sum")

        for col in self.price_columns[self.model_name]:
            last_part = col.split("_")[-1]
            if last_part == "price":
                continue

            df[f"week_price_7d_{last_part}"] = df.groupby(["week", "year"], sort=False)[f"price_{last_part}"].transform(
                "mean"
            )

        df["price_factor_1d"] = df["price_1d"] / df["week_price_1d"]
        df["weekday"] = np.where(df["dayofweek"] < 4, True, False)

        # Ensure weekday factor is maximum 1
        mask = (df["dayofweek"] < 4) & (df["price_factor_1d"] > 1)
        df.loc[mask, "price_factor_1d"] = 1

        # Ensure weekend factor is minimum 1
        mask = (df["dayofweek"] >= 4) & (df["price_factor_1d"] < 1)
        df.loc[mask, "price_factor_1d"] = 1

        df["weekday_factor"] = df.groupby(["week", "year", "weekday"], sort=False)["price_factor_1d"].transform("mean")

        # Multiply 7d price with weekday_factor for weekdays and with price_factor_1d for weekends
        mask = df["dayofweek"] < 4
        df.loc[mask, "price"] = df.loc[mask, "week_price_7d"] * df.loc[mask, "weekday_factor"]

        for col in self.price_columns[self.model_name]:
            last_part = col.split("_")[-1]
            if last_part == "price":
                continue

            df.loc[mask, f"price_{last_part}"] = (
                df.loc[mask, f"week_price_7d_{last_part}"] * df.loc[mask, "weekday_factor"]
            )

        df.loc[~mask, "price"] = df.loc[~mask, "week_price_7d"] * df.loc[~mask, "price_factor_1d"]

        for col in self.price_columns[self.model_name]:
            last_part = col.split("_")[-1]
            if last_part == "price":
                continue

            df.loc[~mask, f"price_{last_part}"] = (
                df.loc[~mask, f"week_price_7d_{last_part}"] * df.loc[~mask, "price_factor_1d"]
            )

        # Ensure sum of week price is still the same
        df["week_price_7d_sum_new"] = df.groupby(["week", "year"], sort=False)["price"].transform("sum")
        df["week_price_7d_sum_diff"] = (df["week_price_7d_sum_new"] - df["week_price_7d_sum"]) / 7
        df["price"] = df["price"] - df["week_price_7d_sum_diff"]

        # df["date_in"] = df["date_in"].dt.strftime("%Y-%m-%d")

        df = df[["date_in"] + self.price_columns[self.model_name]]

        return df

    def country_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.country_code == "gb":
            df[self.price_columns[self.model_name]] *= 0.9
        elif self.country_code in ("fr", "it", "es"):
            df[self.price_columns[self.model_name]] *= 1.1
        return df

    def apply_vat_for_de(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.country_code == "de":
            df[self.price_columns[self.model_name]] *= 0.96
        return df

    def manual_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manually override based on priors (not reflected in data so hard/impossible for model to pick up)"""
        df["month"] = pd.to_datetime(df["date_in"]).dt.month

        # General increases/decreases (multipliers)
        Factor = namedtuple("Factor", ["condition", "factor"])

        multipliers = {
            "has_pool": Factor(True, 1.03),
            "is_detached": Factor(True, 1.02),
            "beach_lake_front": Factor(True, 1.08),
            "apartment": Factor(True, 0.99),
            "aircondition": Factor(True, 1.015),
            "kitchen": Factor(False, 0.8),
            "is_park": Factor(True, 0.99),
        }
        for key, value in multipliers.items():
            if self.input_data[key] == multipliers[key].condition:
                df[self.price_columns[self.model_name]] *= value.factor

        adders = {
            "bbq": 1.5,
            "child_friendly": 2,
            "gym_fitness": 3,
            "parking": 3,
            "garden": 2,
            "pets_allowed": 2,
        }
        for key, value_ in adders.items():
            if self.input_data[key] == True:
                df[self.price_columns[self.model_name]] += value_

        # Special handling of in_ski_area
        if self.input_data["in_ski_area"] is True:
            mask = df["month"].isin([12, 1, 2, 3])
            df.loc[mask, self.price_columns[self.model_name]] *= 1.06

            mask = df["month"].isin([4, 5, 6, 7, 8, 9, 10, 11])
            df.loc[mask, self.price_columns[self.model_name]] *= 0.98

        # The scraped price data has a bias, sometimes price does not include additional costs
        # to compensate for this bias we increase the estimated price a bit
        df[self.price_columns[self.model_name]] *= 1.03

        df = df.drop(columns=["month"])

        # Ensure august weeks are not cheaper than prior weeks
        if self.input_data["country"] == "it":
            df = self.italy_august(df=df)

        # country adjustment
        self.country_adjustments(df=df)

        if self.input_data["country"] == "de":
            df = self.apply_vat_for_de(df=df)

        return df

    @staticmethod
    def ensure_upper_lower(model_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure price_upper > price > price_lower"""

        if model_name == "base_price_recent":
            mask = df["price"] >= df["price_q0.9"]
            df.loc[mask, "price_q0.9"] = df.loc[mask, "price"] + 1

            mask = df["price"] <= df["price_q0.1"]
            df.loc[mask, "price_q0.1"] = df.loc[mask, "price"] - 1
        else:

            mask = df["price"] >= df["price_upper"]
            df.loc[mask, "price_upper"] = df.loc[mask, "price"] + 1

            mask = df["price"] <= df["price_lower"]
            df.loc[mask, "price_lower"] = df.loc[mask, "price"] - 1

        return df

    def align_prices_between_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Allow up to 2.5% increase/decrease of prices per year, based on 2020 prices"""

        df["year"] = df["date_in"].apply(d_year)

        # get predictions
        start_year = 2020
        end_year = df["year"].max()
        predictions_baseline = self.get_model_prediction(specific_year=start_year)
        for year in range(start_year + 1, end_year + 1):
            predictions_second_year = self.get_model_prediction(specific_year=year)

            # allow up to 5% price change per year
            max_change = (year - start_year) * 0.05
            df = self.align(
                base_data=predictions_baseline,
                year_data=predictions_second_year,
                year=year,
                max_change=max_change,
                df=df,
            )

        return df.drop(columns=["year"])

    def align(
        self,
        base_data: pd.DataFrame,
        year_data: pd.DataFrame,
        year: int,
        max_change: float,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        # Calculate average price change
        base_mean = base_data["price"].mean()
        year_mean = year_data["price"].mean()
        change = year_mean / base_mean - 1

        # Dont do anything if average price difference between years is small
        if abs(change) < max_change:
            return df

        sign = change / abs(change)
        new_mean = base_mean * (1 + max_change * sign)
        factor = new_mean / year_mean
        mask = df["year"] == year
        for price_type in self.price_columns[self.model_name]:
            df.loc[mask, price_type] = df.loc[mask, price_type] * factor

        return df

    def load_models(self) -> List[LightGBMRegressionModel]:
        # Load model for the selected country
        file = get_model_location(
            country_code=self.country_code,
            all_models=False,
            model_name=self.model_name,
            version=self.version,
        )
        print(file)
        data = LightGBMRegressionModel.load(file)
        return LightGBMRegressionModel.unserialize(data)

    def get_outliers(self, df: pd.DataFrame) -> Dict[str, str]:
        # outliers = self.model.check_for_outliers(df.iloc[0].to_dict())

        outliers = {}

        return outliers

    def construct_response(self, df: pd.DataFrame, outliers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """using provided data construct dict response as defined in data model"""

        df = df[["date_in"] + self.price_columns[self.model_name]]
        df[self.price_columns[self.model_name]] = df[self.price_columns[self.model_name]].astype(int)

        if outliers is None:
            outliers = {}

        return {
            "version": VERSION,
            "price": df.to_dict("records"),
            "currency": "gbp" if self.country_code == "gb" else "eur",
            "price_method": self.input_data["price_method"],
            "outliers": outliers,
        }

    @staticmethod
    def data_to_df(data: Dict[str, Any], specific_year: Optional[int] = None) -> pd.DataFrame:
        """Returns pd.DataFrame with all static input features and the required date range for
        predictions:

        """
        if specific_year is not None:
            start_date = datetime.date(specific_year, 1, 1)
            end_date = datetime.date(specific_year, 12, 31) + datetime.timedelta(days=1)
        else:
            start_date = datetime.datetime.strptime(data["release_start"], "%Y-%m-%d")
            end_date = datetime.datetime.strptime(data["release_end"], "%Y-%m-%d") + datetime.timedelta(days=1)

        dates = list_of_date(start_date, end_date)  # Adding one day to end_date as end_date is excluded.
        df_date = pd.DataFrame(dates, columns=["date_in"])

        # Construct dataframe with one row from the dict
        df = pd.DataFrame(data=data, index=[0])

        # Contruct artificial key so we can perform cross join
        df["key"] = 0
        df_date["key"] = 0
        df = pd.merge(df, df_date, how="outer", on=["key"])

        # Remove temp key
        df.drop("key", axis=1, inplace=True)

        return df

    def italy_august(self, df: pd.DataFrame) -> pd.DataFrame:
        """BD comments and analysis indicates that certain august weeks should be adjusted compared to prior weeks"""

        df["week"] = df["date_in"].apply(d_week)
        df["year"] = df["date_in"].apply(d_year)

        for weeks in [(29, 30), (30, 31), (32, 33)]:
            mask_first_week = (df["week"] == weeks[0]) & (df["year"] == 2021)
            price_first_week_avg = df.loc[mask_first_week]["price"].mean()
            mask_second_week = (df["week"] == weeks[1]) & (df["year"] == 2021)
            price_second_week_avg = df.loc[mask_second_week]["price"].mean()

            ratio = price_first_week_avg / price_second_week_avg
            if (
                ratio > 1
            ):  # If first week avg price is higher than second week then increase all night prices in second week
                df.loc[mask_second_week, self.price_columns[self.model_name]] *= ratio

        df = df.drop(columns=["week", "year"])

        return df
