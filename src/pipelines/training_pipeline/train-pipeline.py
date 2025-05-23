import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import hopsworks
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

# Modificar _SRCDIR para que apunte a la raíz del proyecto
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from utils import util  # noqa: E402

settings = config.HopsworksSettings(_env_file=f"{_PROJECT_ROOT}/.env")

warnings.filterwarnings("ignore")

logger.info("Setup Hopsworks connection")
# Check if HOPSWORKS_API_KEY env variable is set or if it is set in ~/.env
if settings.HOPSWORKS_API_KEY is not None:
    api_key = settings.HOPSWORKS_API_KEY.get_secret_value()
    os.environ["HOPSWORKS_API_KEY"] = api_key
project = hopsworks.login()
fs = project.get_feature_store()

secrets = hopsworks.get_secrets_api()
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)
country = location["country"]
city = location["city"]
street = location["street"]

logger.info("Retrieve information from feature store")
air_quality_fg = fs.get_feature_group(
    name="air_quality",
    version=1,
)
weather_fg = fs.get_feature_group(
    name="weather",
    version=1,
)

logger.info("Select Features for Training Data")
# Select features for training data.
selected_features = air_quality_fg.select(["pm25"]).join(
    weather_fg.select_all(include_primary_key=False), on=["city"]
)

logger.info("Create Feature View")
# Create a feature view with the selected features.
feature_view = fs.get_or_create_feature_view(
    name="air_quality_fv",
    description="weather features with air quality as the target",
    version=1,
    labels=["pm25"],
    query=selected_features,
)

logger.info("Split the data into train and test sets")
# %%
start_date_test_data = "2025-3-15"
# Convert string to datetime object
test_start = datetime.strptime(start_date_test_data, "%Y-%m-%d")

X_train, X_test, y_train, y_test = feature_view.train_test_split(test_start=test_start)

# Drop the index columns - 'date' (event_time) and 'city' (primary key)

train_features = X_train.drop(["date"], axis="columns")
test_features = X_test.drop(["date"], axis="columns")

logger.info("Train the model")
# Creating an instance of the XGBoost Regressor
xgb_regressor = XGBRegressor()

# Fitting the XGBoost Regressor to the training data
xgb_regressor.fit(train_features, y_train)

logger.info("Predicting PM2.5 values in Test Data")
# Predicting target values on the test set
y_pred = xgb_regressor.predict(test_features)

# Calculating Mean Squared Error (MSE) using sklearn
mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
logger.info(f"MSE: {mse}")
# Calculating R squared using sklearn
r2 = r2_score(y_test.iloc[:, 0], y_pred)
logger.info(f"R squared: {r2}")

logger.info("building final dataframe")
df = y_test
df["predicted_pm25"] = y_pred

df["date"] = X_test["date"]
df = df.sort_values(by=["date"])


# Creating a directory for the model artifacts if it doesn't exist
model_dir = "air_quality_model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
images_dir = model_dir + "/images"
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

file_path = images_dir + "/pm25_hindcast.png"
figure = util.plot_air_quality_forecast(city, street, df, file_path, hindcast=True)

# Plotting feature importances using the plot_importance function from XGBoost
plot_importance(xgb_regressor, max_num_features=4)
feature_importance_path = images_dir + "/feature_importance.png"
plt.savefig(feature_importance_path)


logger.info("Saving model artifacts")
# Saving the XGBoost regressor object as a json file in the model directory
xgb_regressor.save_model(model_dir + "/model.json")

res_dict = {
    "MSE": str(mse),
    "R squared": str(r2),
}

mr = project.get_model_registry()

# Creating a Python model in the model registry named 'air_quality_xgboost_model'

aq_model = mr.python.create_model(
    name="air_quality_xgboost_model",
    metrics=res_dict,
    feature_view=feature_view,
    description="Air Quality (PM2.5) predictor",
)

# Saving the model artifacts to the 'air_quality_model' directory in the model registry
aq_model.save(model_dir)
