# %%
import datetime
import json
import os  # Necesario para os.makedirs
import sys
from pathlib import Path  # Importar Path

import hopsworks
import pandas as pd
from loguru import logger
from xgboost import XGBRegressor

# Modificar _SRCDIR para que apunte a la raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ajustar importaciones
import config  # noqa: E402
from utils import util  # noqa: E402

settings = config.HopsworksSettings(_env_file=f"{_PROJECT_ROOT}/.env")

today = datetime.datetime.now() - datetime.timedelta(5)


logger.info("Setup Hopsworks connection")
project = hopsworks.login()
fs = project.get_feature_store()

secrets = hopsworks.get_secrets_api()
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)
country = location["country"]
city = location["city"]
street = location["street"]

logger.info("Get model from model registry")
mr = project.get_model_registry()

retrieved_model = mr.get_model(
    name="air_quality_xgboost_model",
    version=1,
)

fv = retrieved_model.get_feature_view()

# Download the saved model artifacts to a local directory
saved_model_dir = retrieved_model.download()

# %%
# Loading the XGBoost regressor model and label encoder from the saved model directory
# retrieved_xgboost_model = joblib.load(saved_model_dir + "/xgboost_regressor.pkl")
retrieved_xgboost_model = XGBRegressor()

retrieved_xgboost_model.load_model(saved_model_dir + "/model.json")

logger.info("Retrieve information from feature store")
weather_fg = fs.get_feature_group(
    name="weather",
    version=1,
)
batch_data = weather_fg.filter(weather_fg.date >= today).read()


logger.info("Predicting PM2.5 values in Batch")
batch_data["predicted_pm25"] = retrieved_xgboost_model.predict(
    batch_data[
        [
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        ]
    ]
)


batch_data["street"] = street
batch_data["city"] = city
batch_data["country"] = country
# Fill in the number of days before the date on which you made the forecast (base_date)
batch_data["days_before_forecast_day"] = list(range(1, len(batch_data) + 1))
batch_data = batch_data.sort_values(by=["date"])


pred_file_path_str = f"{_PROJECT_ROOT}/docs/air-quality/assets/img/pm25_forecast.png"
# Asegurarse de que el directorio exista
os.makedirs(Path(pred_file_path_str).parent, exist_ok=True)
figure = util.plot_air_quality_forecast(city, street, batch_data, pred_file_path_str)


# Get or create feature group
monitor_fg = fs.get_or_create_feature_group(
    name="aq_predictions",
    description="Air Quality prediction monitoring",
    version=1,
    primary_key=["city", "street", "date", "days_before_forecast_day"],
    event_time="date",
)


monitor_fg.insert(batch_data, wait=True)


# We will create a hindcast chart for  only the forecasts made 1 day beforehand
monitoring_df = monitor_fg.filter(monitor_fg.days_before_forecast_day == 1).read()

# %%
air_quality_fg = fs.get_feature_group(name="air_quality", version=1)
air_quality_df = air_quality_fg.read()


# %%
outcome_df = air_quality_df[["date", "pm25"]]
preds_df = monitoring_df[["date", "predicted_pm25"]]

hindcast_df = pd.merge(preds_df, outcome_df, on="date")
hindcast_df = hindcast_df.sort_values(by=["date"])

# If there are no outcomes for predictions yet, generate some predictions/outcomes from existing data
if len(hindcast_df) == 0:
    hindcast_df = util.backfill_predictions_for_monitoring(
        weather_fg, air_quality_df, monitor_fg, retrieved_xgboost_model
    )

hindcast_file_path_str = f"{_PROJECT_ROOT}/docs/air-quality/assets/img/pm25_hindcast_1day.png"
# Asegurarse de que el directorio exista
os.makedirs(Path(hindcast_file_path_str).parent, exist_ok=True)
fig = util.plot_air_quality_forecast(
    city, street, hindcast_df, hindcast_file_path_str, hindcast=True
)


dataset_api = project.get_dataset_api()
str_today = today.strftime("%Y-%m-%d")
if not dataset_api.exists("Resources/airquality"):
    dataset_api.mkdir("Resources/airquality")
dataset_api.upload(
    pred_file_path_str, f"Resources/airquality/{city}_{street}_{str_today}", overwrite=True
)
dataset_api.upload(
    hindcast_file_path_str,
    f"Resources/airquality/{city}_{street}_{str_today}",
    overwrite=True,
)

proj_url = project.get_url()
print(f"See images in Hopsworks here: {proj_url}/settings/fb/path/Resources/airquality")
