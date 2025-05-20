"""Feature pipeline for the Hopsworks Air Quality Prediction project"""

import datetime
import json
import os  # For path manipulation
import sys  # For sys.path modification
import warnings

import hopsworks
import pandas as pd
from loguru import logger

# Add the 'src' directory to sys.path
# This allows modules in 'src' (like config.py) and packages in 'src' (like utils)
# to be imported directly when this script is run.
# __file__ is src/pipelines/feature_pipeline/feature-pipeline.py
# os.path.dirname(__file__) is src/pipelines/feature_pipeline
# os.path.join(os.path.dirname(__file__), '..', '..') is src/
_SRCDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRCDIR not in sys.path:
    sys.path.insert(0, _SRCDIR)


import config  # noqa: E402
from utils import util  # noqa: E402

settings = config.HopsworksSettings(_env_file=".env")
warnings.filterwarnings("ignore")

logger.info("Setup Hopsworks connection")
# Set up Hopsworks connection
project = hopsworks.login()
fs = project.get_feature_store()
secrets = hopsworks.get_secrets_api()

# This line will fail if you have not registered the AQICN_API_KEY as a secret in Hopsworks
AQICN_API_KEY = secrets.get_secret("AQICN_API_KEY").value
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)

country = location["country"]
city = location["city"]
street = location["street"]
aqicn_url = location["aqicn_url"]
latitude = location["latitude"]
longitude = location["longitude"]

today = datetime.date.today()
logger.debug(f"Location: {location}")
logger.info("Retrieve information from feature store")
# Retrieve feature groups
air_quality_fg = fs.get_feature_group(
    name="air_quality",
    version=1,
)
weather_fg = fs.get_feature_group(
    name="weather",
    version=1,
)

logger.info("Get today information from AQICN API")
aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, AQICN_API_KEY)

aq_today_df.info()

hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude)
hourly_df = hourly_df.set_index("date")

# We will only make 1 daily prediction, so we will replace the hourly forecasts with a single daily forecast
# We only want the daily weather data, so only get weather at 12:00
daily_df = hourly_df.between_time("11:59", "12:01")
daily_df = daily_df.reset_index()
daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
daily_df["date"] = pd.to_datetime(daily_df["date"])
daily_df["city"] = city

logger.info("insert today data into feature store")
# Insert new data
air_quality_fg.insert(aq_today_df)

# Insert new data
weather_fg.insert(daily_df, wait=True)
