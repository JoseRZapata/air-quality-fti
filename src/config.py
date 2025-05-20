from pathlib import Path  # pragma: no cover
from typing import Literal  # pragma: no cover

from pydantic import SecretStr  # pragma: no cover
from pydantic_settings import BaseSettings, SettingsConfigDict  # pragma: no cover


class HopsworksSettings(BaseSettings):  # pragma: no cover
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    MLFS_DIR: Path = Path(__file__).parent

    # For hopsworks.login(),
    # set as environment variables if they are not already set as env variables
    HOPSWORKS_API_KEY: SecretStr | None = None
    HOPSWORKS_PROJECT: str | None = None
    HOPSWORKS_HOST: str | None = None

    # Air Quality
    AQICN_API_KEY: SecretStr | None = None
    AQICN_COUNTRY: str = "colombia"
    AQICN_CITY: str = "medellin"
    AQICN_STREET: str = "el-poblado"
    AQICN_URL: str = "https://api.waqi.info/feed/@12635"

    # Inference
    RANKING_MODEL_TYPE: Literal["ranking", "llmranking"] = "ranking"
    CUSTOM_HOPSWORKS_INFERENCE_ENV: str = "custom_env_name"
