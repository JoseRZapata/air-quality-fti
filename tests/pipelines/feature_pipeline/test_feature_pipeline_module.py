"""Tests that execute the real linear feature-pipeline script under mocks.

The script runs everything at import time, so it is loaded with
``importlib.util.spec_from_file_location`` + ``exec_module`` while its external
dependencies (config, hopsworks and utils.util functions) are patched.
"""

import datetime
import importlib.util
import json
import sys
import warnings
from collections.abc import Generator
from pathlib import Path
from types import ModuleType
from unittest import mock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "src" / "pipelines" / "feature_pipeline" / "feature-pipeline.py"
MODULE_NAME = "feature_pipeline_script_under_test"

FULL_DAY_HOURS = range(24)
NOON_FILTERED_HOURS = [*range(12), *range(13, 24)]
BETWEEN_START = "11:59"
BETWEEN_END = "12:01"
TEST_CITY = "Test City"


def build_hourly_dataframe(hours: range | list[int]) -> pd.DataFrame:
    """Build a real hourly weather DataFrame for today.

    Args:
        hours: Hours of the day to include as rows.

    Returns:
        pd.DataFrame: Hourly weather data with a ``date`` column.
    """
    today = datetime.date.today()
    dates = [datetime.datetime.combine(today, datetime.time(hour=hour)) for hour in hours]
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_mean": [20.0 + index / 10 for index in range(len(dates))],
            "precipitation_sum": [0.1 * index for index in range(len(dates))],
            "wind_speed_10m_max": [5.0 + index / 5 for index in range(len(dates))],
            "wind_direction_10m_dominant": [180 + 5 * index for index in range(len(dates))],
        }
    )


@pytest.fixture
def secrets_data() -> dict[str, str]:
    """Provide fake Hopsworks secrets values.

    Returns:
        dict[str, str]: Mapping of secret keys to their string values.
    """
    return {
        "AQICN_API_KEY": "fake-aqicn-key",  # pragma: allowlist secret
        "SENSOR_LOCATION_JSON": json.dumps(
            {
                "country": "Test Country",
                "city": TEST_CITY,
                "street": "Test Street",
                "aqicn_url": "http://test-aqicn-url.com",
                "latitude": 40.7128,
                "longitude": -74.0060,
            }
        ),
    }


@pytest.fixture
def aq_today_df(secrets_data: dict[str, str]) -> pd.DataFrame:
    """Build the DataFrame returned by the mocked get_pm25.

    Args:
        secrets_data: Fake secrets used to derive location fields.

    Returns:
        pd.DataFrame: Single-row air quality measurement for today.
    """
    location = json.loads(secrets_data["SENSOR_LOCATION_JSON"])
    return pd.DataFrame(
        {
            "pm25": [15.2],
            "country": [location["country"]],
            "city": [location["city"]],
            "street": [location["street"]],
            "date": [datetime.date.today()],
            "url": [location["aqicn_url"]],
        }
    )


@pytest.fixture
def pipeline_context(
    secrets_data: dict[str, str], aq_today_df: pd.DataFrame
) -> Generator[dict[str, object], None, None]:
    """Patch every external dependency and execute the real script once.

    Args:
        secrets_data: Fake secrets served by the mocked secrets API.
        aq_today_df: DataFrame returned by the mocked get_pm25.

    Yields:
        Generator[dict[str, object], None, None]: Mocks plus the hourly_df used,
        cleaned from ``sys.modules`` afterwards.
    """
    hourly_df = build_hourly_dataframe(FULL_DAY_HOURS)
    settings_instance = mock.MagicMock()

    project = mock.MagicMock()
    fs = project.get_feature_store.return_value
    air_quality_fg = mock.MagicMock()
    weather_fg = mock.MagicMock()
    fs.get_feature_group.side_effect = lambda name, version: (
        air_quality_fg if name == "air_quality" else weather_fg
    )

    secrets_api = mock.MagicMock()

    def get_secret(key: str) -> mock.MagicMock:
        secret = mock.MagicMock()
        secret.value = secrets_data[key]
        return secret

    secrets_api.get_secret.side_effect = get_secret

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with (
        warnings.catch_warnings(),
        mock.patch("config.HopsworksSettings", return_value=settings_instance),
        mock.patch("hopsworks.login", return_value=project) as mock_login,
        mock.patch("hopsworks.get_secrets_api", return_value=secrets_api),
        mock.patch("utils.util.get_pm25", return_value=aq_today_df) as mock_get_pm25,
        mock.patch(
            "utils.util.get_hourly_weather_forecast", return_value=hourly_df
        ) as mock_get_hourly_weather,
    ):
        try:
            sys.modules[MODULE_NAME] = module
            spec.loader.exec_module(module)
            yield {
                "module": module,
                "login": mock_login,
                "fs": fs,
                "air_quality_fg": air_quality_fg,
                "weather_fg": weather_fg,
                "get_pm25": mock_get_pm25,
                "get_hourly_weather": mock_get_hourly_weather,
                "hourly_df": hourly_df,
            }
        finally:
            sys.modules.pop(MODULE_NAME, None)


def _expected_daily_df(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the expected daily DataFrame from an hourly DataFrame.

    Args:
        hourly_df: Hourly weather data returned by the mocked forecast function.

    Returns:
        pd.DataFrame: Daily row at noon with transformed dates and city column.
    """
    daily_df = hourly_df.set_index("date").between_time(BETWEEN_START, BETWEEN_END).reset_index()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["city"] = TEST_CITY
    return daily_df


def test_full_script_execution(
    pipeline_context: dict[str, object], secrets_data: dict[str, str], aq_today_df: pd.DataFrame
) -> None:
    """Verify the whole linear script flow when executed under mocks.

    Checks login, feature group retrieval, keyword-only get_pm25 arguments,
    weather forecast arguments, and both feature group insert calls including
    the transformations applied to daily_df.

    Args:
        pipeline_context: Mocks captured while executing the module.
        secrets_data: Fake secrets used to derive expected values.
        aq_today_df: DataFrame expected to be inserted into air_quality_fg.
    """
    context = pipeline_context
    module: ModuleType = context["module"]
    location = json.loads(secrets_data["SENSOR_LOCATION_JSON"])

    assert module is not None
    context["login"].assert_called_once_with()

    fs: mock.MagicMock = context["fs"]
    assert fs.get_feature_group.call_args_list[0].kwargs == {
        "name": "air_quality",
        "version": 1,
    }
    assert fs.get_feature_group.call_args_list[1].kwargs == {"name": "weather", "version": 1}

    get_pm25: mock.MagicMock = context["get_pm25"]
    get_pm25.assert_called_once_with(
        location["aqicn_url"],
        country=location["country"],
        city=location["city"],
        street=location["street"],
        day=datetime.date.today(),
        AQI_API_KEY=secrets_data["AQICN_API_KEY"],  # pragma: allowlist secret
    )
    assert list(get_pm25.call_args.args) == [location["aqicn_url"]]
    assert set(get_pm25.call_args.kwargs) == {"country", "city", "street", "day", "AQI_API_KEY"}

    get_hourly_weather: mock.MagicMock = context["get_hourly_weather"]
    get_hourly_weather.assert_called_once_with(
        location["city"], location["latitude"], location["longitude"]
    )

    air_quality_fg: mock.MagicMock = context["air_quality_fg"]
    air_quality_fg.insert.assert_called_once()
    pd.testing.assert_frame_equal(air_quality_fg.insert.call_args.args[0], aq_today_df)

    weather_fg: mock.MagicMock = context["weather_fg"]
    weather_fg.insert.assert_called_once()
    inserted_daily_df = weather_fg.insert.call_args.args[0]
    assert weather_fg.insert.call_args.kwargs == {"wait": True}

    expected_daily_df = _expected_daily_df(context["hourly_df"])
    pd.testing.assert_frame_equal(inserted_daily_df, expected_daily_df)
    assert (inserted_daily_df["city"] == TEST_CITY).all()
    assert len(inserted_daily_df) == 1
    assert inserted_daily_df["date"].iloc[0] == pd.Timestamp(datetime.date.today())
    assert pd.api.types.is_datetime64_any_dtype(inserted_daily_df["date"])


def test_script_execution_without_noon_rows(
    secrets_data: dict[str, str], aq_today_df: pd.DataFrame
) -> None:
    """Verify the script flow when no hourly row falls between 11:59-12:01.

    The daily DataFrame must be empty but still carry the city column and a
    proper datetime date column before being inserted with wait=True.

    Args:
        secrets_data: Fake secrets served by the mocked secrets API.
        aq_today_df: DataFrame returned by the mocked get_pm25.
    """
    hourly_df = build_hourly_dataframe(NOON_FILTERED_HOURS)

    project = mock.MagicMock()
    fs = project.get_feature_store.return_value
    air_quality_fg = mock.MagicMock()
    weather_fg = mock.MagicMock()
    fs.get_feature_group.side_effect = lambda name, version: (
        air_quality_fg if name == "air_quality" else weather_fg
    )

    secrets_api = mock.MagicMock()

    def get_secret(key: str) -> mock.MagicMock:
        secret = mock.MagicMock()
        secret.value = secrets_data[key]
        return secret

    secrets_api.get_secret.side_effect = get_secret

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with (
        warnings.catch_warnings(),
        mock.patch("config.HopsworksSettings", return_value=mock.MagicMock()),
        mock.patch("hopsworks.login", return_value=project),
        mock.patch("hopsworks.get_secrets_api", return_value=secrets_api),
        mock.patch("utils.util.get_pm25", return_value=aq_today_df),
        mock.patch("utils.util.get_hourly_weather_forecast", return_value=hourly_df),
    ):
        try:
            sys.modules[MODULE_NAME] = module
            spec.loader.exec_module(module)
            inserted_daily_df = weather_fg.insert.call_args.args[0]
        finally:
            sys.modules.pop(MODULE_NAME, None)

    assert weather_fg.insert.call_args.kwargs == {"wait": True}
    assert len(inserted_daily_df) == 0
    assert "city" in inserted_daily_df.columns
    assert pd.api.types.is_datetime64_any_dtype(inserted_daily_df["date"])
