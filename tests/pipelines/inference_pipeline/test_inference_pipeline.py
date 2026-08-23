"""Tests for the linear inference pipeline script executed via importlib."""

import contextlib
import datetime
import importlib.util
import json
import sys
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pipelines"
    / "inference_pipeline"
    / "inference-pipeline.py"
)

FEATURE_COLUMNS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]

LOCATION_JSON = json.dumps({"country": "colombia", "city": "medellin", "street": "el-poblado"})
FAKE_URL = "https://fake.hopsworks"
EXPECTED_PLOT_CALLS = 2
EXPECTED_UPLOAD_CALLS = 2
N_ROWS = 4


class _ComparableDate:
    def __ge__(self, other: object) -> mock.MagicMock:
        return mock.MagicMock(name="filter_expression")


def _make_batch_data(n_rows: int = N_ROWS) -> pd.DataFrame:
    base_date = datetime.datetime.now() - datetime.timedelta(2)
    return pd.DataFrame(
        {
            "date": [base_date + datetime.timedelta(days=i) for i in range(n_rows)],
            **{column: np.linspace(10.0, 30.0, n_rows) for column in FEATURE_COLUMNS},
        }
    )


def _make_monitoring_df(batch_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": batch_data["date"].tolist(),
            "predicted_pm25": [5.0] * len(batch_data),
        }
    )


def _make_air_quality_df(batch_data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": batch_data["date"].tolist(),
            "pm25": [7.0] * len(batch_data),
        }
    )


def _build_hopsworks_mocks(
    tmp_path: Path,
    batch_data: pd.DataFrame,
    monitoring_df: pd.DataFrame,
    air_quality_df: pd.DataFrame,
) -> dict[str, Any]:
    last_version = 7

    feature_view = mock.MagicMock(name="feature_view")

    retrieved_model = mock.MagicMock(name="retrieved_model")
    retrieved_model.get_feature_view.return_value = feature_view
    saved_model_dir = tmp_path / "fake_model_dir"
    saved_model_dir.mkdir()
    retrieved_model.download.return_value = str(saved_model_dir)

    model_registry = mock.MagicMock(name="model_registry")
    registered_model = mock.MagicMock()
    registered_model.version = last_version
    model_registry.get_models.return_value = [registered_model]
    model_registry.get_model.return_value = retrieved_model

    weather_fg = mock.MagicMock(name="weather_fg")
    weather_fg.date = _ComparableDate()
    weather_fg.filter.return_value.read.return_value = batch_data.copy()

    monitor_fg = mock.MagicMock(name="monitor_fg")
    monitor_fg.filter.return_value.read.return_value = monitoring_df.copy()

    air_quality_fg = mock.MagicMock(name="air_quality_fg")
    air_quality_fg.read.return_value = air_quality_df.copy()

    feature_store = mock.MagicMock(name="feature_store")
    feature_store.get_feature_group.side_effect = lambda name, version: {
        "weather": weather_fg,
        "air_quality": air_quality_fg,
    }[name]
    feature_store.get_or_create_feature_group.return_value = monitor_fg

    secrets_api = mock.MagicMock(name="secrets_api")
    secret = mock.MagicMock()
    secret.value = LOCATION_JSON
    secrets_api.get_secret.return_value = secret

    dataset_api = mock.MagicMock(name="dataset_api")
    dataset_api.exists.return_value = True

    project = mock.MagicMock(name="project")
    project.get_feature_store.return_value = feature_store
    project.get_model_registry.return_value = model_registry
    project.get_dataset_api.return_value = dataset_api
    project.get_url.return_value = FAKE_URL

    return {
        "project": project,
        "secrets_api": secrets_api,
        "model_registry": model_registry,
        "retrieved_model": retrieved_model,
        "weather_fg": weather_fg,
        "monitor_fg": monitor_fg,
        "air_quality_fg": air_quality_fg,
        "dataset_api": dataset_api,
        "last_version": last_version,
    }


def _exec_module(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _fake_xgboost_module() -> Iterator[None]:
    original: ModuleType | None = sys.modules.get("xgboost")
    sys.modules["xgboost"] = mock.MagicMock(name="xgboost")
    try:
        yield
    finally:
        if original is None:
            sys.modules.pop("xgboost", None)
        else:
            sys.modules["xgboost"] = original


@pytest.fixture
def clean_sys_modules() -> Generator[Callable[[str], None], None, None]:
    created_names: list[str] = []

    def register(module_name: str) -> None:
        created_names.append(module_name)

    yield register

    for name in created_names:
        sys.modules.pop(name, None)


def _run_pipeline(module_name: str, tmp_path: Path, frames: dict[str, Any]) -> mock.Mock:
    mocks = _build_hopsworks_mocks(
        tmp_path,
        frames["batch_data"],
        frames["monitoring_df"],
        frames["air_quality_df"],
    )
    project = mocks["project"]
    model_instance = mock.MagicMock(name="xgb_model_instance")
    predictions = np.linspace(1.0, 9.0, len(frames["batch_data"]))
    model_instance.predict.return_value = predictions

    with (
        _fake_xgboost_module(),
        mock.patch("config.HopsworksSettings"),
        mock.patch("hopsworks.login", return_value=project),
        mock.patch("hopsworks.get_secrets_api", return_value=mocks["secrets_api"]),
        mock.patch("xgboost.XGBRegressor", return_value=model_instance),
        mock.patch("utils.util.plot_air_quality_forecast") as plot_mock,
        mock.patch("utils.util.backfill_predictions_for_monitoring") as backfill_mock,
        mock.patch("os.makedirs") as makedirs_mock,
        mock.patch("builtins.print") as print_mock,
    ):
        if frames.get("backfill_return") is not None:
            backfill_mock.return_value = frames["backfill_return"]
        module = _exec_module(module_name)

    context = mock.Mock()
    context.mocks = mocks
    context.model_instance = model_instance
    context.plot_mock = plot_mock
    context.backfill_mock = backfill_mock
    context.makedirs_mock = makedirs_mock
    context.print_mock = print_mock
    context.module = module
    return context


def test_happy_path_runs_full_pipeline_without_backfill(
    tmp_path: Path, clean_sys_modules: Callable[[str], None]
) -> None:
    """Full pipeline runs, inserts predictions and uploads both charts."""
    clean_sys_modules("inference_pipeline_happy_path")
    batch_data = _make_batch_data()
    context = _run_pipeline(
        "inference_pipeline_happy_path",
        tmp_path,
        {
            "batch_data": batch_data,
            "monitoring_df": _make_monitoring_df(batch_data),
            "air_quality_df": _make_air_quality_df(batch_data),
        },
    )

    mocks = context.mocks
    mocks["model_registry"].get_models.assert_called_once_with(name="air_quality_xgboost_model")
    mocks["model_registry"].get_model.assert_called_once_with(
        name="air_quality_xgboost_model", version=mocks["last_version"]
    )

    saved_model_dir = mocks["retrieved_model"].download.return_value
    context.model_instance.load_model.assert_called_once_with(f"{saved_model_dir}/model.json")

    inserted_frame = mocks["monitor_fg"].insert.call_args[0][0]
    for column in ("predicted_pm25", "street", "city", "country"):
        assert column in inserted_frame.columns
    assert inserted_frame["days_before_forecast_day"].tolist() == list(range(1, 5))
    assert (
        inserted_frame["predicted_pm25"].tolist()
        == context.module.batch_data["predicted_pm25"].tolist()
    )

    assert context.plot_mock.call_count == EXPECTED_PLOT_CALLS
    first_call = context.plot_mock.call_args_list[0]
    second_call = context.plot_mock.call_args_list[1]
    assert second_call.kwargs.get("hindcast") is True
    assert not first_call.kwargs.get("hindcast", False)

    assert mocks["dataset_api"].upload.call_count == EXPECTED_UPLOAD_CALLS
    for upload_call in mocks["dataset_api"].upload.call_args_list:
        assert upload_call.kwargs.get("overwrite") is True

    context.backfill_mock.assert_not_called()

    printed_output = "".join(str(call_arg) for call_arg in context.print_mock.call_args[0])
    assert FAKE_URL in printed_output


def test_empty_hindcast_triggers_backfill(
    tmp_path: Path, clean_sys_modules: Callable[[str], None]
) -> None:
    """When no outcomes match predictions, backfill is used for the hindcast."""
    clean_sys_modules("inference_pipeline_backfill")
    batch_data = _make_batch_data()
    stale_monitoring_df = pd.DataFrame(
        {
            "date": [datetime.datetime(2020, 1, 1)],
            "predicted_pm25": [5.0],
        }
    )
    air_quality_df = pd.DataFrame(
        {
            "date": [datetime.datetime(2021, 6, 15)],
            "pm25": [7.0],
        }
    )
    context = _run_pipeline(
        "inference_pipeline_backfill",
        tmp_path,
        {
            "batch_data": batch_data,
            "monitoring_df": stale_monitoring_df,
            "air_quality_df": air_quality_df,
            "backfill_return": pd.DataFrame(
                {
                    "date": [batch_data["date"].iloc[0]],
                    "predicted_pm25": [42.0],
                }
            ),
        },
    )

    mocks = context.mocks
    backfill_return = context.backfill_mock.return_value
    context.backfill_mock.assert_called_once_with(
        mocks["weather_fg"],
        mocks["air_quality_fg"].read.return_value,
        mocks["monitor_fg"],
        context.model_instance,
    )

    assert context.plot_mock.call_count == EXPECTED_PLOT_CALLS
    hindcast_call = context.plot_mock.call_args_list[1]
    assert hindcast_call.kwargs.get("hindcast") is True
    assert hindcast_call.args[2].equals(backfill_return)

    mocks["dataset_api"].mkdir.assert_not_called()
