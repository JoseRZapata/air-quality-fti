import os
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HopsworksSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    MLFS_DIR: Path = Path(__file__).parent

    # For hopsworks.login(), set as environment variables if they are not already set as env variables
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

    def model_post_init(self, __context):
        """Runs after the model is initialized."""
        print("HopsworksSettings initialized!")
        if os.getenv("HOPSWORKS_API_KEY") == None:
            if self.HOPSWORKS_API_KEY is not None:
                os.environ["HOPSWORKS_API_KEY"] = (
                    self.HOPSWORKS_API_KEY.get_secret_value()
                )
        if os.getenv("HOPSWORKS_PROJECT") == None:
            if self.HOPSWORKS_PROJECT is not None:
                os.environ["HOPSWORKS_PROJECT"] = self.HOPSWORKS_PROJECT
        if os.getenv("HOPSWORKS_HOST") == None:
            if self.HOPSWORKS_HOST is not None:
                os.environ["HOPSWORKS_HOST"] = self.HOPSWORKS_HOST
