import datetime
from pathlib import Path
from typing import Any

import hopsworks.client.exceptions as hopsworks_exceptions
import hsfs
import matplotlib.pyplot as plt
import openmeteo_requests
import pandas as pd
import requests  # type: ignore
import requests_cache
from geopy.geocoders import Nominatim
from hopsworks.project import Project  # Added import
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from hsml.model_registry import ModelRegistry
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from retry_requests import retry


def get_historical_weather(
    city: str, start_date: str, end_date: str, latitude: float, longitude: float
) -> pd.DataFrame:
    """Gets historical weather data for a given city and date range from Open-Meteo.

    Args:
        city (str): The name of the city.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        latitude (float): The latitude of the city.
        longitude (float): The longitude of the city.

    Returns:
        pd.DataFrame: A DataFrame containing the historical weather data with columns
            for date, temperature_2m_mean, precipitation_sum, wind_speed_10m_max,
            wind_direction_10m_dominant, and city.
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
        ],
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(3).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s"),
            end=pd.to_datetime(daily.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
    }
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

    daily_dataframe = pd.DataFrame(data=daily_data)
    daily_dataframe = daily_dataframe.dropna()
    daily_dataframe["city"] = city
    return daily_dataframe


def get_hourly_weather_forecast(city: str, latitude: float, longitude: float) -> pd.DataFrame:
    """Gets hourly weather forecast data for a given city from Open-Meteo.

    Args:
        city (str): The name of the city.
        latitude (float): The latitude of the city.
        longitude (float): The longitude of the city.

    Returns:
        pd.DataFrame: A DataFrame containing the hourly weather forecast data with
            columns for date, temperature_2m_mean, precipitation_sum,
            wind_speed_10m_max, and wind_direction_10m_dominant.
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s"),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    hourly_data["temperature_2m_mean"] = hourly_temperature_2m
    hourly_data["precipitation_sum"] = hourly_precipitation
    hourly_data["wind_speed_10m_max"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m_dominant"] = hourly_wind_direction_10m

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe = hourly_dataframe.dropna()
    return hourly_dataframe


def get_city_coordinates(city_name: str) -> tuple[float, float]:
    """Gets the latitude and longitude of a city using Nominatim.

    Args:
        city_name (str): The name of the city.

    Returns:
        tuple[float, float]: A tuple containing the latitude and longitude of the city,
            rounded to two decimal places.
    """
    geolocator = Nominatim(user_agent="MyApp")
    city_location = geolocator.geocode(city_name)

    latitude = round(city_location.latitude, 2)
    longitude = round(city_location.longitude, 2)

    return latitude, longitude


def trigger_request(url: str, timeout: int = 10) -> dict[Any, Any]:
    """Makes a GET request to the specified URL and returns the JSON response.

    Args:
        url (str): The URL to make the GET request to.
        timeout (int, optional): The timeout for the request in seconds.
            Defaults to 10.

    Returns:
        dict[Any, Any]: The JSON response from the GET request.

    Raises:
        TypeError: If the JSON response is not a dictionary.
        requests.exceptions.RequestException: If the request fails or returns
            a non-OK status code.
    """
    response = requests.get(url, timeout=timeout)
    if response.status_code == requests.codes.ok:
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise TypeError("Bad JSON")
        data: dict[Any, Any] = data_json
    else:
        print(f"Failed to retrieve data. Status Code: {response.status_code}")
        raise requests.exceptions.RequestException("HTTP error")

    return data


def get_pm25(  # noqa: PLR0913
    aqicn_url: str,
    country: str,
    city: str,
    street: str,
    day: datetime.date,
    AQI_API_KEY: str,
) -> pd.DataFrame:
    """Retrieves PM2.5 air quality data from the AQICN API for a specific location and day.

    It first tries the base AQICN URL. If the station is unknown, it retries
    with more specific URLs including country/street and then country/city/street.

    Args:
        aqicn_url (str): The base URL of the AQICN API for the sensor.
        country (str): The country name.
        city (str): The city name.
        street (str): The street name.
        day (datetime.date): The date for which to get the air quality data.
        AQI_API_KEY (str): The API key for the AQICN API.

    Returns:
        pd.DataFrame: A DataFrame containing the PM2.5 data, country, city, street,
            date, and the URL used. Columns include 'pm25', 'country', 'city',
            'street', 'date', 'url'.

    Raises:
        requests.exceptions.RequestException: If the API request fails or returns
            an error status after all retry attempts.
    """
    url = f"{aqicn_url}/?token={AQI_API_KEY}"

    data = trigger_request(url)

    if data["data"] == "Unknown station":
        url1 = f"https://api.waqi.info/feed/{country}/{street}/?token={AQI_API_KEY}"
        data = trigger_request(url1)

    if data["data"] == "Unknown station":
        url2 = f"https://api.waqi.info/feed/{country}/{city}/{street}/?token={AQI_API_KEY}"
        data = trigger_request(url2)

    if data["status"] == "ok":
        aqi_data = data["data"]
        aq_today_df = pd.DataFrame()
        aq_today_df["pm25"] = [aqi_data["iaqi"].get("pm25", {}).get("v", None)]
        aq_today_df["pm25"] = aq_today_df["pm25"].astype("float32")

        aq_today_df["country"] = country
        aq_today_df["city"] = city
        aq_today_df["street"] = street
        aq_today_df["date"] = day
        aq_today_df["date"] = pd.to_datetime(aq_today_df["date"])
        aq_today_df["url"] = aqicn_url
    else:
        print(
            "Error: There may be an incorrect URL for your Sensor or it is not contactable "
            "right now. The API response does not contain data.  Error message:",
            data["data"],
        )
        raise requests.exceptions.RequestException(data["data"])

    return aq_today_df


def plot_air_quality_forecast(
    city: str, street: str, df: pd.DataFrame, file_path: str, hindcast: bool = False
) -> Figure:
    """Plots air quality forecast data and saves it to a file.

    The plot displays predicted PM2.5 values on a logarithmic scale. It can
    optionally include actual PM2.5 values for hindcasting. Air Quality Index
    categories are shown as colored horizontal spans.

    Args:
        city (str): The name of the city for the plot title.
        street (str): The name of the street for the plot title.
        df (pd.DataFrame): DataFrame containing the data to plot. Expected columns
            include 'date' and 'predicted_pm25'. If `hindcast` is True,
            it also expects a 'pm25' column for actual values.
        file_path (str): The full path where the plot image will be saved.
        hindcast (bool, optional): If True, plots actual PM2.5 data alongside
            predictions. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    day_data = pd.to_datetime(df["date"]).dt.date
    ax.plot(
        day_data,
        df["predicted_pm25"],
        label="Predicted PM2.5",
        color="red",
        linewidth=2,
        marker="o",
        markersize=5,
        markerfacecolor="blue",
    )

    ax.set_yscale("log")
    ax.set_yticks([0, 10, 25, 50, 100, 250, 500])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=1)

    ax.set_xlabel("Date")
    ax.set_title(f"PM2.5 Predicted (Logarithmic Scale) for {city}, {street}")
    ax.set_ylabel("PM2.5")

    colors_list = ["green", "yellow", "orange", "red", "purple", "darkred"]
    labels_list = [
        "Good",
        "Moderate",
        "Unhealthy for Some",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    ranges_list = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
    for color_item, (start, end) in zip(colors_list, ranges_list, strict=False):
        ax.axhspan(start, end, color=color_item, alpha=0.3)

    patches = [
        Patch(
            color=colors_list[i], label=f"{labels_list[i]}: {ranges_list[i][0]}-{ranges_list[i][1]}"
        )
        for i in range(len(colors_list))
    ]
    legend1 = ax.legend(
        handles=patches,
        loc="upper right",
        title="Air Quality Categories",
        fontsize="x-small",
    )

    ANNOTATED_VALUES = 11
    if len(df.index) > ANNOTATED_VALUES:
        every_x_tick = len(df.index) / 10
        ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast:
        ax.plot(
            day_data,
            df["pm25"],
            label="Actual PM2.5",
            color="black",
            linewidth=2,
            marker="^",
            markersize=5,
            markerfacecolor="grey",
        )
        current_handles, current_labels = ax.get_legend_handles_labels()
        if legend1:
            ax.add_artist(legend1)
        ax.legend(current_handles, current_labels, loc="upper left", fontsize="x-small")

    plt.tight_layout()
    plt.savefig(file_path)
    return fig


def delete_feature_groups(fs: FeatureStore, name: str) -> None:
    """Deletes all versions of a feature group with the given name.

    Args:
        fs (hsfs.feature_store.FeatureStore): The Hopsworks feature store object.
        name (str): The name of the feature group to delete.
    """
    try:
        for fg in fs.get_feature_groups(name):
            fg.delete()
            print(f"Deleted {fg.name}/{fg.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature group found")


def delete_feature_views(fs: FeatureStore, name: str) -> None:
    """Deletes all versions of a feature view with the given name.

    Args:
        fs (hsfs.feature_store.FeatureStore): The Hopsworks feature store object.
        name (str): The name of the feature view to delete.
    """
    try:
        for fv in fs.get_feature_views(name):
            fv.delete()
            print(f"Deleted {fv.name}/{fv.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature view found")


def delete_models(mr: ModelRegistry, name: str) -> None:
    """Deletes all versions of a model with the given name from the model registry.

    Args:
        mr (ModelRegistry): The Hopsworks model registry object.
        name (str): The name of the model to delete.
    """
    models_list = mr.get_models(name)
    if not models_list:
        print(f"No {name} model found")
    for model_item in models_list:
        model_item.delete()
        print(f"Deleted model {model_item.name}/{model_item.version}")


def delete_secrets(proj: Project, name: str) -> None:  # Changed hopsworks.Project to Project
    """Deletes a secret with the given name from the project.

    Args:
        proj (Project): The Hopsworks project object.
        name (str): The name of the secret to delete.
    """
    s_api = proj.get_secrets_api()
    try:
        s_api.delete(name)
        print(f"Deleted secret {name}")
    except hopsworks_exceptions.RestAPIError:
        print(f"No {name} secret found")


def purge_project(proj: Project) -> None:
    """Purges all feature data, models, and secrets from the project.

    Args:
        proj (Project): The Hopsworks project object.
    """
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()

    delete_feature_views(fs, "air_quality_fv")
    delete_feature_groups(fs, "air_quality")
    delete_feature_groups(fs, "weather")
    delete_feature_groups(fs, "aq_predictions")
    delete_models(mr, "air_quality_xgboost_model")
    delete_secrets(proj, "SENSOR_LOCATION_JSON")


def check_file_path(file_path: str) -> None:
    """Checks if a file exists at the given path and prints a status message.

    Args:
        file_path (str): The path to the file to check.
    """
    my_file = Path(file_path)
    if not my_file.is_file():
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")


def backfill_predictions_for_monitoring(
    weather_fg: FeatureGroup,
    air_quality_df: pd.DataFrame,
    monitor_fg: FeatureGroup,
    model: Any,
) -> pd.DataFrame:
    r"""Generates hindcast predictions and inserts them into a monitoring feature group.

    This function reads the latest weather data, makes predictions using the provided
    model, merges these predictions with actual air quality data, and inserts
    the results into the monitoring feature group. It also returns the
    DataFrame containing the hindcast data.

    Args:
        weather_fg (hsfs.feature_group.FeatureGroup): The feature group containing
            weather data. Expected to have a 'date' column and feature columns
            used by the model.
        air_quality_df (pd.DataFrame): DataFrame containing actual air quality data,
            including 'date' and 'pm25' columns.
        monitor_fg (hsfs.feature_group.FeatureGroup): The feature group where
            the hindcast predictions will be inserted.
        model (Any): The trained model object to use for making predictions.
            It must have a \`predict\` method compatible with the features from
            \`weather_fg\`.

    Returns:
        pd.DataFrame: A DataFrame containing the merged weather features,
            predicted PM2.5, actual PM2.5, and other relevant information
            for hindcasting.
    """
    features_df = weather_fg.read()
    features_df = features_df.sort_values(by=["date"], ascending=True)
    features_df = features_df.tail(10)
    features_df["predicted_pm25"] = model.predict(
        features_df[
            [
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
            ]
        ]
    )
    df_merged = pd.merge(
        features_df, air_quality_df[["date", "pm25", "street", "country"]], on="date"
    )
    df_merged["days_before_forecast_day"] = 1
    hindcast_df = df_merged.copy()
    df_to_insert = df_merged.drop("pm25", axis=1)
    monitor_fg.insert(df_to_insert, write_options={"wait_for_job": True})
    return hindcast_df
