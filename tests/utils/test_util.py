"""Unit tests for the utility functions in src.utils.util."""

import datetime
from collections.abc import Generator  # Añadir importación
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import requests  # type: ignore
from matplotlib.figure import Figure

# Importar las funciones del módulo util
# Necesitamos ajustar el path si es necesario o asegurar que PYTHONPATH esté configurado
# Asumiendo que pytest se ejecuta desde la raíz del proyecto y src está en pythonpath
from src.utils import util


@pytest.fixture
def mock_geolocator() -> Generator[mock.MagicMock, None, None]:  # Corregir tipo de retorno
    """Fixture to mock geopy.geocoders.Nominatim."""
    with mock.patch("src.utils.util.Nominatim") as mock_nominatim:
        mock_instance = mock_nominatim.return_value
        mock_location = mock.MagicMock()
        mock_location.latitude = 40.7128
        mock_location.longitude = -74.0060
        mock_instance.geocode.return_value = mock_location
        yield mock_instance


def test_get_city_coordinates(mock_geolocator: mock.MagicMock) -> None:
    """Test get_city_coordinates function.

    Args:
        mock_geolocator (mock.MagicMock): Mocked Nominatim instance.
    """
    city_name = "New York"
    EXPECTED_LATITUDE = 40.71  # Rounded from mock_location.latitude = 40.7128
    EXPECTED_LONGITUDE = -74.01  # Rounded from mock_location.longitude = -74.0060

    latitude, longitude = util.get_city_coordinates(city_name)

    mock_geolocator.geocode.assert_called_once_with(city_name)
    assert latitude == EXPECTED_LATITUDE
    assert longitude == EXPECTED_LONGITUDE


@mock.patch("src.utils.util.requests.get")
def test_trigger_request_success(mock_requests_get: mock.MagicMock) -> None:
    """Test trigger_request function for a successful request.

    Args:
        mock_requests_get (mock.MagicMock): Mocked requests.get.
    """
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    expected_json = {"data": "success"}
    mock_response.json.return_value = expected_json
    mock_requests_get.return_value = mock_response

    url = "http://fakeurl.com/api"
    result = util.trigger_request(url)

    mock_requests_get.assert_called_once_with(url, timeout=10)
    assert result == expected_json


@mock.patch("src.utils.util.requests.get")
def test_trigger_request_http_error(mock_requests_get: mock.MagicMock) -> None:
    """Test trigger_request function for an HTTP error.

    Args:
        mock_requests_get (mock.MagicMock): Mocked requests.get.
    """
    mock_response = mock.MagicMock()
    mock_response.status_code = 404
    mock_requests_get.return_value = mock_response

    url = "http://fakeurl.com/api"
    with pytest.raises(requests.exceptions.RequestException, match="HTTP error"):
        util.trigger_request(url)
    mock_requests_get.assert_called_once_with(url, timeout=10)


@mock.patch("src.utils.util.requests.get")
def test_trigger_request_bad_json(mock_requests_get: mock.MagicMock) -> None:
    """Test trigger_request function for a response with bad JSON.

    Args:
        mock_requests_get (mock.MagicMock): Mocked requests.get.
    """
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = "not a dict"  # Invalid JSON for the function
    mock_requests_get.return_value = mock_response

    url = "http://fakeurl.com/api"
    with pytest.raises(TypeError, match="Bad JSON"):
        util.trigger_request(url)
    mock_requests_get.assert_called_once_with(url, timeout=10)


@mock.patch("src.utils.util.Path.is_file")
def test_check_file_path_file_exists(
    mock_is_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test check_file_path when the file exists.

    Args:
        mock_is_file (mock.MagicMock): Mocked Path.is_file.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    mock_is_file.return_value = True
    file_path = "fake/existing/file.txt"
    util.check_file_path(file_path)
    captured = capsys.readouterr()
    assert f"File successfully found at the path: {file_path}" in captured.out
    mock_is_file.assert_called_once()


@mock.patch("src.utils.util.Path.is_file")
def test_check_file_path_file_not_exists(
    mock_is_file: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test check_file_path when the file does not exist.

    Args:
        mock_is_file (mock.MagicMock): Mocked Path.is_file.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    mock_is_file.return_value = False
    file_path = "fake/non_existing/file.txt"
    util.check_file_path(file_path)
    captured = capsys.readouterr()
    assert f"Error. File not found at the path: {file_path}" in captured.out
    mock_is_file.assert_called_once()


@mock.patch("src.utils.util.trigger_request")
def test_get_pm25_success(mock_trigger_request: mock.MagicMock) -> None:
    """Test get_pm25 for a successful data retrieval.

    Args:
        mock_trigger_request (mock.MagicMock): Mocked trigger_request.
    """
    api_key = "test_api_key"  # pragma: allowlist secret
    base_url = "http://baseaqiurl.com/feed/station"
    country, city, street = "Wonderland", "Teapot", "Madhatter St"
    test_date = datetime.date(2023, 1, 1)
    EXPECTED_PM25_VALUE = 25.5

    mock_trigger_request.return_value = {
        "status": "ok",
        "data": {
            "iaqi": {"pm25": {"v": EXPECTED_PM25_VALUE}},
            # ... other data ...
        },
    }

    df = util.get_pm25(base_url, country, city, street, test_date, api_key)

    mock_trigger_request.assert_called_once_with(f"{base_url}/?token={api_key}")
    assert not df.empty
    assert "pm25" in df.columns
    assert df["pm25"].iloc[0] == EXPECTED_PM25_VALUE


@mock.patch("src.utils.util.trigger_request")
def test_get_pm25_unknown_station_retry_success(
    mock_trigger_request: mock.MagicMock,
) -> None:
    """Test get_pm25 with retries due to 'Unknown station'.

    Args:
        mock_trigger_request (mock.MagicMock): Mocked trigger_request.
    """
    api_key = "test_api_key"  # pragma: allowlist secret
    base_url = "http://baseaqiurl.com/feed/station_unknown"
    country, city, street = "Oz", "Emerald City", "Yellow Brick Rd"
    test_date = datetime.date(2023, 1, 2)
    EXPECTED_PM25_VALUE_RETRY = 10.0
    EXPECTED_CALL_COUNT = 2

    # Simulate "Unknown station" on first call, then success on second
    mock_trigger_request.side_effect = [
        {"status": "ok", "data": "Unknown station"},
        {
            "status": "ok",
            "data": {
                "iaqi": {"pm25": {"v": EXPECTED_PM25_VALUE_RETRY}},
            },
        },
    ]

    df = util.get_pm25(base_url, country, city, street, test_date, api_key)

    assert mock_trigger_request.call_count == EXPECTED_CALL_COUNT
    calls = [
        mock.call(f"{base_url}/?token={api_key}"),
        mock.call(f"https://api.waqi.info/feed/{country}/{street}/?token={api_key}"),
    ]
    mock_trigger_request.assert_has_calls(calls)

    assert not df.empty
    assert df["pm25"].iloc[0] == EXPECTED_PM25_VALUE_RETRY
    assert df["country"].iloc[0] == country


@mock.patch("src.utils.util.trigger_request")
def test_get_pm25_all_retries_fail(mock_trigger_request: mock.MagicMock) -> None:
    """Test get_pm25 when all retry attempts fail.

    Args:
        mock_trigger_request (mock.MagicMock): Mocked trigger_request.
    """
    api_key = "test_api_key"  # pragma: allowlist secret
    base_url = "http://baseaqiurl.com/feed/station_always_unknown"
    country, city, street = "Neverland", "Lagoon", "Pirate Cove"
    test_date = datetime.date(2023, 1, 3)
    EXPECTED_CALL_COUNT = 3

    mock_trigger_request.side_effect = [
        {"status": "ok", "data": "Unknown station"},
        {"status": "ok", "data": "Unknown station"},
        {"status": "error", "data": "Sensor offline"},  # Final failure
    ]

    with pytest.raises(requests.exceptions.RequestException, match="Sensor offline"):
        util.get_pm25(base_url, country, city, street, test_date, api_key)

    assert mock_trigger_request.call_count == EXPECTED_CALL_COUNT
    calls = [
        mock.call(f"{base_url}/?token={api_key}"),
        mock.call(f"https://api.waqi.info/feed/{country}/{street}/?token={api_key}"),
        mock.call(f"https://api.waqi.info/feed/{country}/{city}/{street}/?token={api_key}"),
    ]
    mock_trigger_request.assert_has_calls(calls)


# --- Pruebas para funciones que interactúan con Open-Meteo ---


@pytest.fixture
def mock_openmeteo_client() -> Generator[mock.MagicMock, None, None]:  # Corregir tipo de retorno
    """Fixture to mock openmeteo_requests.Client."""
    with mock.patch("src.utils.util.openmeteo_requests.Client") as mock_client_class:
        mock_client_instance = mock_client_class.return_value
        mock_response = mock.MagicMock()

        # Mockear atributos de la respuesta de OpenMeteo
        mock_response.Latitude.return_value = 52.52
        mock_response.Longitude.return_value = 13.41
        mock_response.Elevation.return_value = 34.0
        mock_response.Timezone.return_value = "Europe/Berlin"
        mock_response.TimezoneAbbreviation.return_value = "CEST"
        mock_response.UtcOffsetSeconds.return_value = 7200

        # Mockear Daily/Hourly y Variables
        mock_daily_or_hourly = mock.MagicMock()
        mock_daily_or_hourly.Time.return_value = 1609459200  # 2021-01-01
        mock_daily_or_hourly.TimeEnd.return_value = 1609545600  # 2021-01-02
        mock_daily_or_hourly.Interval.return_value = 86400  # 1 day for daily

        mock_variable = mock.MagicMock()
        mock_variable.ValuesAsNumpy.return_value = pd.Series(
            [
                10.0,
                12.0,
            ]
        ).to_numpy()  # Ejemplo de datos

        mock_daily_or_hourly.Variables.return_value = mock_variable
        mock_response.Daily.return_value = mock_daily_or_hourly
        mock_response.Hourly.return_value = (
            mock_daily_or_hourly  # Reutilizar para hourly con ajustes
        )

        mock_client_instance.weather_api.return_value = [mock_response]
        yield mock_client_instance


@mock.patch("src.utils.util.requests_cache.CachedSession")
@mock.patch("src.utils.util.retry")
def test_get_historical_weather(
    mock_retry_session: mock.MagicMock,
    mock_cached_session: mock.MagicMock,
    mock_openmeteo_client: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test get_historical_weather function.

    Args:
        mock_retry_session (mock.MagicMock): Mocked retry session.
        mock_cached_session (mock.MagicMock): Mocked CachedSession.
        mock_openmeteo_client (mock.MagicMock): Mocked OpenMeteo client.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    city = "Berlin"
    start_date = "2023-01-01"
    end_date = "2023-01-01"  # Solo un día para simplificar el mock de datos
    latitude = 52.52
    longitude = 13.41
    EXPECTED_DF_LEN = 1
    EXPECTED_TEMP = 10.0
    EXPECTED_PRECIP = 5.0
    EXPECTED_WIND_SPEED = 15.0
    EXPECTED_WIND_DIR = 180.0

    # Ajustar el mock de OpenMeteo para que devuelva un solo valor para un día
    mock_response = mock_openmeteo_client.weather_api.return_value[0]
    mock_daily = mock_response.Daily.return_value
    mock_daily.TimeEnd.return_value = 1609459200 + 86400  # Un día después
    mock_variable = mock_daily.Variables.return_value
    # Necesitamos 4 variables para daily
    mock_variable.ValuesAsNumpy.side_effect = [
        pd.Series([EXPECTED_TEMP]).to_numpy(),  # temp
        pd.Series([EXPECTED_PRECIP]).to_numpy(),  # precip
        pd.Series([EXPECTED_WIND_SPEED]).to_numpy(),  # wind_speed
        pd.Series([EXPECTED_WIND_DIR]).to_numpy(),  # wind_dir
    ]

    df = util.get_historical_weather(city, start_date, end_date, latitude, longitude)

    mock_cached_session.assert_called_once_with(".cache", expire_after=-1)
    mock_retry_session.assert_called_once()
    mock_openmeteo_client.weather_api.assert_called_once()

    assert not df.empty
    assert "temperature_2m_mean" in df.columns
    assert df["city"].iloc[0] == city
    assert len(df) == EXPECTED_DF_LEN  # Un día de datos
    assert df["temperature_2m_mean"].iloc[0] == EXPECTED_TEMP
    assert df["precipitation_sum"].iloc[0] == EXPECTED_PRECIP
    assert df["wind_speed_10m_max"].iloc[0] == EXPECTED_WIND_SPEED
    assert df["wind_direction_10m_dominant"].iloc[0] == EXPECTED_WIND_DIR

    captured = capsys.readouterr()
    assert "Coordinates 52.52°N 13.41°E" in captured.out


@mock.patch("src.utils.util.requests_cache.CachedSession")
@mock.patch("src.utils.util.retry")
def test_get_hourly_weather_forecast(
    mock_retry_session: mock.MagicMock,
    mock_cached_session: mock.MagicMock,
    mock_openmeteo_client: mock.MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test get_hourly_weather_forecast function.

    Args:
        mock_retry_session (mock.MagicMock): Mocked retry session.
        mock_cached_session (mock.MagicMock): Mocked CachedSession.
        mock_openmeteo_client (mock.MagicMock): Mocked OpenMeteo client.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    city = "Paris"
    latitude = 48.8566
    longitude = 2.3522
    EXPECTED_DF_LEN = 2
    EXPECTED_TEMP_MEAN_0 = 8.0
    EXPECTED_TEMP_MEAN_1 = 8.5
    EXPECTED_PRECIP_SUM_0 = 0.1
    EXPECTED_PRECIP_SUM_1 = 0.2
    EXPECTED_WIND_SPEED_0 = 10.0
    EXPECTED_WIND_SPEED_1 = 11.0
    EXPECTED_WIND_DIR_0 = 90.0
    EXPECTED_WIND_DIR_1 = 95.0

    # Ajustar el mock de OpenMeteo para datos horarios
    mock_response = mock_openmeteo_client.weather_api.return_value[0]
    mock_hourly = mock_response.Hourly.return_value
    mock_hourly.Time.return_value = 1609459200  # 2021-01-01 00:00
    # Usar EXPECTED_DF_LEN para calcular TimeEnd
    mock_hourly.TimeEnd.return_value = 1609459200 + 3600 * EXPECTED_DF_LEN
    mock_hourly.Interval.return_value = 3600  # 1 hora

    mock_variable = mock_hourly.Variables.return_value
    mock_variable.ValuesAsNumpy.side_effect = [
        pd.Series([EXPECTED_TEMP_MEAN_0, EXPECTED_TEMP_MEAN_1]).to_numpy(),  # temp
        pd.Series([EXPECTED_PRECIP_SUM_0, EXPECTED_PRECIP_SUM_1]).to_numpy(),  # precip
        pd.Series([EXPECTED_WIND_SPEED_0, EXPECTED_WIND_SPEED_1]).to_numpy(),  # wind_speed
        pd.Series([EXPECTED_WIND_DIR_0, EXPECTED_WIND_DIR_1]).to_numpy(),  # wind_dir
    ]

    df = util.get_hourly_weather_forecast(city, latitude, longitude)

    mock_cached_session.assert_called_once_with(".cache", expire_after=3600)
    mock_retry_session.assert_called_once()
    mock_openmeteo_client.weather_api.assert_called_once()

    assert not df.empty
    assert "temperature_2m_mean" in df.columns
    assert len(df) == EXPECTED_DF_LEN  # Dos horas de datos
    assert df["temperature_2m_mean"].iloc[0] == EXPECTED_TEMP_MEAN_0
    assert df["precipitation_sum"].iloc[1] == EXPECTED_PRECIP_SUM_1

    captured = capsys.readouterr()
    assert "Coordinates 52.52°N 13.41°E" in captured.out  # Usando el mock general


# --- Pruebas para plot_air_quality_forecast ---


@mock.patch("src.utils.util.plt")
def test_plot_air_quality_forecast(mock_plt: mock.MagicMock, tmp_path: Path) -> None:
    """Test plot_air_quality_forecast function.

    Args:
        mock_plt (mock.MagicMock): Mocked matplotlib.pyplot.
        tmp_path (Path): Pytest fixture for a temporary directory.
    """
    city = "TestCity"
    street = "TestStreet"
    file_name = "test_plot.png"
    file_path = tmp_path / file_name
    # Number of Air Quality Index categories plotted with axhspan
    NUM_AQI_CATEGORIES = 6

    data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
        "predicted_pm25": [50.0, 60.0],
        "pm25": [45.0, 65.0],  # Para hindcast
    }
    df = pd.DataFrame(data)

    # Mockear la figura y el eje devueltos por subplots
    mock_fig = mock.MagicMock(spec=Figure)
    mock_ax = mock.MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    # Mockear get_legend_handles_labels para evitar errores si se llama
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    fig_result = util.plot_air_quality_forecast(city, street, df, str(file_path), hindcast=True)

    assert fig_result == mock_fig
    mock_plt.subplots.assert_called_once_with(figsize=(10, 6))
    assert mock_ax.plot.call_count >= 1  # Al menos una llamada para predicciones
    mock_ax.set_xlabel.assert_called_with("Date")
    mock_ax.set_title.assert_called_with(
        f"PM2.5 Predicted (Logarithmic Scale) for {city}, {street}"
    )
    mock_ax.set_ylabel.assert_called_with("PM2.5")
    mock_plt.savefig.assert_called_once_with(str(file_path))
    mock_plt.tight_layout.assert_called_once()
    mock_plt.xticks.assert_called_with(rotation=45)

    # Verificar que se llamó a axhspan para las categorías de calidad del aire
    assert mock_ax.axhspan.call_count == NUM_AQI_CATEGORIES

    # Verificar que se llamó a plot para los datos reales (hindcast=True)
    # La segunda llamada a plot (índice 1) debería ser para los datos reales
    hindcast_plot_called = False
    for call_args in mock_ax.plot.call_args_list:
        if "Actual PM2.5" in call_args[1].get("label", ""):
            hindcast_plot_called = True
            break
    assert hindcast_plot_called


# --- Pruebas para funciones de Hopsworks (requieren mocks más complejos) ---
# Estas son más complejas de mockear completamente sin una instancia de Hopsworks
# o mocks muy detallados de la librería hsfs.


@pytest.fixture
def mock_fs() -> mock.MagicMock:
    """Fixture to mock hsfs.feature_store.FeatureStore."""
    return mock.MagicMock(spec=util.FeatureStore)


@pytest.fixture
def mock_mr() -> mock.MagicMock:
    """Fixture to mock hsml.model_registry.ModelRegistry."""
    return mock.MagicMock(spec=util.ModelRegistry)


@pytest.fixture
def mock_project(mock_fs: mock.MagicMock, mock_mr: mock.MagicMock) -> mock.MagicMock:
    """Fixture to mock hopsworks.project.Project."""
    mock_proj = mock.MagicMock()  # Eliminado spec=util.Project temporalmente
    mock_proj.get_feature_store.return_value = mock_fs  # Usar la fixture inyectada
    mock_proj.get_model_registry.return_value = mock_mr  # Usar la fixture inyectada
    mock_proj.get_secrets_api.return_value = mock.MagicMock()
    return mock_proj


def test_delete_feature_groups(mock_fs: mock.MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
    """Test delete_feature_groups function.

    Args:
        mock_fs (mock.MagicMock): Mocked FeatureStore.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    mock_fg1 = mock.MagicMock()
    mock_fg1.name = "test_fg"
    mock_fg1.version = 1
    mock_fg2 = mock.MagicMock()
    mock_fg2.name = "test_fg"
    mock_fg2.version = 2

    mock_fs.get_feature_groups.return_value = [mock_fg1, mock_fg2]

    util.delete_feature_groups(mock_fs, "test_fg")

    mock_fs.get_feature_groups.assert_called_once_with("test_fg")
    mock_fg1.delete.assert_called_once()
    mock_fg2.delete.assert_called_once()
    captured = capsys.readouterr()
    assert "Deleted test_fg/1" in captured.out
    assert "Deleted test_fg/2" in captured.out


def test_delete_feature_groups_not_found(
    mock_fs: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test delete_feature_groups when no feature group is found.

    Args:
        mock_fs (mock.MagicMock): Mocked FeatureStore.
        capsys (pytest.CaptureFixture[str]): Pytest fixture to capture stdout/stderr.
    """
    mock_fs.get_feature_groups.side_effect = util.hsfs.client.exceptions.RestAPIError(
        mock.MagicMock(), mock.MagicMock()
    )
    util.delete_feature_groups(mock_fs, "non_existent_fg")
    captured = capsys.readouterr()
    assert "No non_existent_fg feature group found" in captured.out


# Se pueden añadir pruebas similares para delete_feature_views, delete_models, delete_secrets


@mock.patch("src.utils.util.delete_feature_views")
@mock.patch("src.utils.util.delete_feature_groups")
@mock.patch("src.utils.util.delete_models")
@mock.patch("src.utils.util.delete_secrets")
def test_purge_project(
    mock_delete_secrets: mock.MagicMock,
    mock_delete_models: mock.MagicMock,
    mock_delete_fg: mock.MagicMock,
    mock_delete_fv: mock.MagicMock,
    mock_project: mock.MagicMock,  # Usa el fixture mock_project
) -> None:
    """Test purge_project function.

    Args:
        mock_delete_secrets (mock.MagicMock): Mocked delete_secrets.
        mock_delete_models (mock.MagicMock): Mocked delete_models.
        mock_delete_fg (mock.MagicMock): Mocked delete_feature_groups.
        mock_delete_fv (mock.MagicMock): Mocked delete_feature_views.
        mock_project (mock.MagicMock): Mocked Project instance.
    """
    # Extraer los mocks de fs y mr del mock_project para aserciones
    mock_fs_instance = mock_project.get_feature_store()
    mock_mr_instance = mock_project.get_model_registry()
    # Expected number of calls to delete_feature_groups
    # For "air_quality", "weather", "aq_predictions"
    EXPECTED_DELETE_FG_COUNT = 3

    util.purge_project(mock_project)

    mock_delete_fv.assert_called_once_with(mock_fs_instance, "air_quality_fv")
    mock_delete_fg.assert_any_call(mock_fs_instance, "air_quality")
    mock_delete_fg.assert_any_call(mock_fs_instance, "weather")
    mock_delete_fg.assert_any_call(mock_fs_instance, "aq_predictions")
    assert mock_delete_fg.call_count == EXPECTED_DELETE_FG_COUNT
    mock_delete_models.assert_called_once_with(mock_mr_instance, "air_quality_xgboost_model")
    mock_delete_secrets.assert_called_once_with(mock_project, "SENSOR_LOCATION_JSON")


@pytest.fixture
def mock_feature_group() -> mock.MagicMock:
    """Fixture to mock hsfs.feature_group.FeatureGroup."""
    fg = mock.MagicMock(spec=util.FeatureGroup)
    # Simular el método read() devolviendo un DataFrame simple
    data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "temperature_2m_mean": [10, 12, 11],
        "precipitation_sum": [1, 0, 2],
        "wind_speed_10m_max": [5, 6, 7],
        "wind_direction_10m_dominant": [180, 190, 170],
    }
    fg.read.return_value = pd.DataFrame(data)
    return fg


@pytest.fixture
def mock_model() -> mock.MagicMock:
    """Fixture to mock a generic model with a predict method."""
    model = mock.MagicMock()
    model.predict.return_value = pd.Series([25.0, 30.0, 28.0])  # Ejemplo de predicciones
    return model


def test_backfill_predictions_for_monitoring(  # noqa: PLR0915
    mock_feature_group: mock.MagicMock,  # weather_fg
    mock_model: mock.MagicMock,
) -> None:
    """Test backfill_predictions_for_monitoring function.

    Args:
        mock_feature_group (mock.MagicMock): Mocked weather FeatureGroup.
        mock_model (mock.MagicMock): Mocked model.
    """
    # Based on the length of the input mock data for weather_fg and air_quality_df
    EXPECTED_DF_LEN = 3
    # From mock_model.predict.return_value for iloc[0]
    PREDICTED_PM25_VAL_0 = 25.0
    # From air_quality_df setup for iloc[0]
    ACTUAL_PM25_VAL_0 = 22.0
    # days_before_forecast_day is idx + 1, so for iloc[0], idx is 0
    EXPECTED_DAYS_BEFORE_0 = 1

    # Data for air_quality_df, using constants for clarity
    ACTUAL_PM25_VAL_1 = 33.0  # For iloc[1]
    ACTUAL_PM25_VAL_2 = 25.0  # For iloc[2]

    # Crear un DataFrame de air_quality_df de prueba
    aq_data = {
        "date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "pm25": [ACTUAL_PM25_VAL_0, ACTUAL_PM25_VAL_1, ACTUAL_PM25_VAL_2],
        "street": ["StreetA", "StreetA", "StreetA"],
        "country": ["CountryX", "CountryX", "CountryX"],
    }
    air_quality_df = pd.DataFrame(aq_data)

    # Crear un mock para monitor_fg
    monitor_fg_mock = mock.MagicMock(spec=util.FeatureGroup)

    # Ajustar el mock_model.predict para que coincida con el número de filas
    # These are other predicted values if needed for other assertions
    PREDICTED_PM25_VAL_1 = 30.0
    PREDICTED_PM25_VAL_2 = 28.0
    mock_model.predict.return_value = pd.Series(
        [
            PREDICTED_PM25_VAL_0,
            PREDICTED_PM25_VAL_1,
            PREDICTED_PM25_VAL_2,
        ]
    )

    hindcast_df = util.backfill_predictions_for_monitoring(
        weather_fg=mock_feature_group,
        air_quality_df=air_quality_df,
        monitor_fg=monitor_fg_mock,
        model=mock_model,
    )

    mock_feature_group.read.assert_called_once()
    # Verificar que predict fue llamado con las columnas correctas
    # La llamada real a predict se hace sobre un DataFrame, así que verificamos el contenido esperado
    pd.testing.assert_frame_equal(
        mock_model.predict.call_args[0][0],
        mock_feature_group.read.return_value[
            [
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
            ]
        ],
    )

    monitor_fg_mock.insert.assert_called_once()
    # La inserción se hace sin la columna 'pm25'
    df_to_insert_arg = monitor_fg_mock.insert.call_args[0][0]
    assert "pm25" not in df_to_insert_arg.columns
    assert "predicted_pm25" in df_to_insert_arg.columns

    assert not hindcast_df.empty
    assert "predicted_pm25" in hindcast_df.columns
    assert "pm25" in hindcast_df.columns  # pm25 real está en hindcast_df
    assert len(hindcast_df) == EXPECTED_DF_LEN  # Basado en los datos de prueba
    assert hindcast_df["predicted_pm25"].iloc[0] == PREDICTED_PM25_VAL_0
    assert hindcast_df["pm25"].iloc[0] == ACTUAL_PM25_VAL_0
    assert hindcast_df["days_before_forecast_day"].iloc[0] == EXPECTED_DAYS_BEFORE_0

    def test_delete_feature_views(
        mock_fs: mock.MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test delete_feature_views function."""
        mock_fv1 = mock.MagicMock()
        mock_fv1.name = "test_fv"
        mock_fv1.version = 1
        mock_fv2 = mock.MagicMock()
        mock_fv2.name = "test_fv"
        mock_fv2.version = 2

        mock_fs.get_feature_views.return_value = [mock_fv1, mock_fv2]

        util.delete_feature_views(mock_fs, "test_fv")

        mock_fs.get_feature_views.assert_called_once_with("test_fv")
        mock_fv1.delete.assert_called_once()
        mock_fv2.delete.assert_called_once()
        captured = capsys.readouterr()
        assert "Deleted test_fv/1" in captured.out
        assert "Deleted test_fv/2" in captured.out

    def test_delete_feature_views_not_found(
        mock_fs: mock.MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test delete_feature_views when no feature view is found."""
        mock_fs.get_feature_views.side_effect = util.hsfs.client.exceptions.RestAPIError(
            mock.MagicMock(), mock.MagicMock()
        )
        util.delete_feature_views(mock_fs, "non_existent_fv")
        captured = capsys.readouterr()
        assert "No non_existent_fv feature view found" in captured.out

    def test_delete_models_found(
        mock_mr: mock.MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test delete_models when models are found."""
        mock_model1 = mock.MagicMock()
        mock_model1.name = "test_model"
        mock_model1.version = 1
        mock_model2 = mock.MagicMock()
        mock_model2.name = "test_model"
        mock_model2.version = 2
        mock_mr.get_models.return_value = [mock_model1, mock_model2]

        util.delete_models(mock_mr, "test_model")

        mock_mr.get_models.assert_called_once_with("test_model")
        mock_model1.delete.assert_called_once()
        mock_model2.delete.assert_called_once()
        captured = capsys.readouterr()
        assert "Deleted model test_model/1" in captured.out
        assert "Deleted model test_model/2" in captured.out

    def test_delete_models_not_found(
        mock_mr: mock.MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test delete_models when no models are found."""
        mock_mr.get_models.return_value = []
        util.delete_models(mock_mr, "non_existent_model")
        captured = capsys.readouterr()
        assert "No non_existent_model model found" in captured.out
