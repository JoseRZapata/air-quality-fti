"""Tests for the linear training pipeline script executed via importlib."""

import contextlib
import datetime
import importlib.util
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pipelines"
    / "training_pipeline"
    / "train-pipeline.py"
)

FEATURE_COLUMNS = [
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]

LOCATION_JSON = json.dumps({"country": "colombia", "city": "medellin", "street": "el-poblado"})
MODEL_DIR_NAME = "air_quality_model"
PROJECT_SRC_DIR = (Path(__file__).resolve().parents[3] / "src").resolve()
FAKE_API_KEY = "fake-hopsworks-api-key"  # pragma: allowlist secret
EXPECTED_FEATURE_GROUPS = 2
N_ROWS = 10


def _make_split_frames(
    n_rows: int = N_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_date = datetime.datetime(2025, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(n_rows)]
    x_train = pd.DataFrame(
        {
            "date": dates[: n_rows // 2],
            **{column: np.linspace(10.0, 20.0, n_rows // 2) for column in FEATURE_COLUMNS},
        }
    )
    x_test = pd.DataFrame(
        {
            "date": dates[n_rows // 2 :],
            **{column: np.linspace(20.0, 30.0, n_rows - n_rows // 2) for column in FEATURE_COLUMNS},
        }
    )
    y_train = pd.DataFrame({"pm25": np.linspace(5.0, 8.0, n_rows // 2)})
    y_test = pd.DataFrame({"pm25": np.linspace(6.0, 9.0, n_rows - n_rows // 2)})
    return x_train, x_test, y_train, y_test


def _build_hopsworks_mocks(
    split_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> dict[str, Any]:
    x_train, x_test, y_train, y_test = split_frames

    feature_view = mock.MagicMock(name="feature_view")
    feature_view.train_test_split.return_value = (x_train, x_test, y_train, y_test)

    air_quality_fg = mock.MagicMock(name="air_quality_fg")
    weather_fg = mock.MagicMock(name="weather_fg")

    feature_store = mock.MagicMock(name="feature_store")
    feature_store.get_feature_group.side_effect = lambda name, version: {
        "air_quality": air_quality_fg,
        "weather": weather_fg,
    }[name]
    feature_store.get_or_create_feature_view.return_value = feature_view

    secrets_api = mock.MagicMock(name="secrets_api")
    secret = mock.MagicMock()
    secret.value = LOCATION_JSON
    secrets_api.get_secret.return_value = secret

    model_registry = mock.MagicMock(name="model_registry")

    project = mock.MagicMock(name="project")
    project.get_feature_store.return_value = feature_store
    project.get_model_registry.return_value = model_registry

    return {
        "project": project,
        "secrets_api": secrets_api,
        "model_registry": model_registry,
        "feature_store": feature_store,
        "air_quality_fg": air_quality_fg,
        "weather_fg": weather_fg,
        "feature_view": feature_view,
    }


@contextlib.contextmanager
def _fake_xgboost_module(model_instance: mock.MagicMock) -> Iterator[Any]:
    fake_module: Any = types.ModuleType("xgboost")
    fake_module.XGBRegressor = mock.MagicMock(return_value=model_instance)
    fake_module.plot_importance = mock.MagicMock()
    original = sys.modules.get("xgboost")
    sys.modules["xgboost"] = fake_module
    try:
        yield fake_module
    finally:
        if original is None:
            sys.modules.pop("xgboost", None)
        else:
            sys.modules["xgboost"] = original


def _is_project_src_entry(entry: str | None) -> bool:
    return Path(entry or ".").resolve() == PROJECT_SRC_DIR


def _exec_module(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_sys_path = sys.path[:]
    sys.path = [entry for entry in sys.path if not _is_project_src_entry(entry)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


def _run_pipeline(
    module_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    api_key: SecretStr | None = None,
) -> mock.Mock:
    split_frames = _make_split_frames()
    mocks = _build_hopsworks_mocks(split_frames)

    model_instance = mock.MagicMock(name="xgb_model_instance")
    test_rows = len(split_frames[1])
    model_instance.predict.return_value = np.linspace(5.5, 8.5, test_rows)

    monkeypatch.chdir(tmp_path)
    if api_key is not None:
        monkeypatch.setenv("HOPSWORKS_API_KEY", "")

    with (
        _fake_xgboost_module(model_instance) as fake_xgb,
        mock.patch("config.HopsworksSettings") as settings_cls,
        mock.patch("hopsworks.login", return_value=mocks["project"]) as login_mock,
        mock.patch(
            "hopsworks.get_secrets_api", return_value=mocks["secrets_api"]
        ) as secrets_api_mock,
        mock.patch("matplotlib.pyplot.savefig") as savefig_mock,
        mock.patch("utils.util.plot_air_quality_forecast") as plot_mock,
    ):
        settings_cls.return_value.HOPSWORKS_API_KEY = api_key
        module = _exec_module(module_name)

    context = mock.Mock()
    context.mocks = mocks
    context.model_instance = model_instance
    context.fake_xgb = fake_xgb
    context.settings_cls = settings_cls
    context.login_mock = login_mock
    context.secrets_api_mock = secrets_api_mock
    context.savefig_mock = savefig_mock
    context.plot_mock = plot_mock
    context.split_frames = split_frames
    context.module = module
    return context


def test_happy_path_trains_and_registers_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full training pipeline runs and registers the model in Hopsworks."""
    module_name = "training_pipeline_happy_path"
    context = _run_pipeline(module_name, tmp_path, monkeypatch)

    mocks = context.mocks

    context.login_mock.assert_called_once_with()

    mocks["project"].get_feature_store.assert_called_once_with()
    assert mocks["feature_store"].get_feature_group.call_count == EXPECTED_FEATURE_GROUPS
    mocks["feature_store"].get_feature_group.assert_any_call(name="air_quality", version=1)
    mocks["feature_store"].get_feature_group.assert_any_call(name="weather", version=1)

    feature_view_kwargs = mocks["feature_store"].get_or_create_feature_view.call_args.kwargs
    assert feature_view_kwargs["name"] == "air_quality_fv"
    assert feature_view_kwargs["version"] == 1
    assert feature_view_kwargs["labels"] == ["pm25"]

    _, _, _, y_test = context.split_frames
    train_features = context.model_instance.fit.call_args[0][0]
    test_features = context.model_instance.predict.call_args[0][0]
    assert list(train_features.columns) == FEATURE_COLUMNS
    assert list(test_features.columns) == FEATURE_COLUMNS
    assert len(train_features) + len(test_features) == N_ROWS
    context.model_instance.predict.assert_called_once_with(test_features)

    context.model_instance.save_model.assert_called_once_with(f"{MODEL_DIR_NAME}/model.json")

    create_call = mocks["model_registry"].python.create_model
    create_call.assert_called_once()
    create_kwargs = create_call.call_args.kwargs
    assert create_kwargs["name"] == "air_quality_xgboost_model"
    assert set(create_kwargs["metrics"]) == {"MSE", "R squared"}
    assert create_kwargs["feature_view"] is mocks["feature_view"]
    created_model = create_call.return_value
    created_model.save.assert_called_once_with(MODEL_DIR_NAME)

    plot_args = context.plot_mock.call_args[0]
    assert plot_args[3] == f"{MODEL_DIR_NAME}/images/pm25_hindcast.png"
    assert context.plot_mock.call_args.kwargs.get("hindcast") is True
    assert (
        plot_args[2]
        .sort_values("date")
        .reset_index(drop=True)
        .equals(
            y_test.assign(predicted_pm25=context.model_instance.predict.return_value)[
                ["pm25", "predicted_pm25", "date"]
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )
    )

    context.fake_xgb.plot_importance.assert_called_once_with(
        context.model_instance, max_num_features=4
    )
    context.savefig_mock.assert_called_once_with(f"{MODEL_DIR_NAME}/images/feature_importance.png")

    images_dir = tmp_path / MODEL_DIR_NAME / "images"
    assert images_dir.is_dir()


def test_api_key_from_settings_is_exported_to_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured HOPSWORKS_API_KEY is exported to the process environment."""
    module_name = "training_pipeline_api_key"
    api_key = SecretStr(FAKE_API_KEY)
    context = _run_pipeline(module_name, tmp_path, monkeypatch, api_key=api_key)

    context.settings_cls.assert_called_once()
    assert context.module.os.environ["HOPSWORKS_API_KEY"] == FAKE_API_KEY
