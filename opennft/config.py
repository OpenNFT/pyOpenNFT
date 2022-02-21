# -*- coding: utf-8 -*-

from typing import NamedTuple, Optional
import os
from pathlib import Path

import appdirs
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .constants import APP_NAME, ENVVAR_PREFIX
from .errors import OpenNftError
from .logging import configure_logger


ROOT_PATH = Path(__file__).parent
PKG_CONFIG_DIR = ROOT_PATH / 'configs'

SITE_CONFIG_DIR = Path(appdirs.site_config_dir(APP_NAME))
USER_CONFIG_DIR = Path(appdirs.user_config_dir(APP_NAME))

APP_CONFIG_ENVVAR = ENVVAR_PREFIX + '_CONFIG'
LOG_CONFIG_ENVVAR = ENVVAR_PREFIX + '_LOG_CONFIG'

APP_CONFIG_FILE_NAME = f'{APP_NAME}.config.yaml'
LOG_CONFIG_FILE_NAME = f'{APP_NAME}.logging.yaml'


class ConfigError(OpenNftError):
    pass


class LoadConfigError(ConfigError):
    pass


class UpdateConfigError(ConfigError):
    pass


class ConfigFileDoesNotExistError(ConfigError):
    pass


class ConfigValueError(ConfigError):
    pass


class ConfigPaths(NamedTuple):
    """Ordered config paths
    """

    default_path: Path  # default config in the app package
    site_path: Path  # site config in /etc/<appname>/
    user_path: Path  # user config in ~/.config/<appname>/


app_config_paths = ConfigPaths(
    default_path=PKG_CONFIG_DIR / APP_CONFIG_FILE_NAME,
    site_path=SITE_CONFIG_DIR / APP_CONFIG_FILE_NAME,
    user_path=USER_CONFIG_DIR / APP_CONFIG_FILE_NAME,
)

log_config_paths = ConfigPaths(
    default_path=PKG_CONFIG_DIR / LOG_CONFIG_FILE_NAME,
    site_path=SITE_CONFIG_DIR / LOG_CONFIG_FILE_NAME,
    user_path=USER_CONFIG_DIR / LOG_CONFIG_FILE_NAME,
)


def update_config(cfg: DictConfig, config_path: Path, label: str, not_exist_error: bool = False) -> DictConfig:
    """Update the existing config object from new config file

    Parameters
    ----------
    cfg: DictConfig
        Configuration object to update.
    config_path : Path
        Config file path from which to update the config.
    label : str
        The label for the configuration to add in __config_paths__ meta info
    not_exist_error : bool
        If True raise ``ConfigFileDoesNotExistError`` if the config file does not exist.

    Returns
    -------
    config : DictConfig
        Updated configuration object

    Raises
    ------
    ConfigFileDoesNotExistError : If ``not_exist_error`` is True and the config file does not exist.
    UpdateConfigError : If an error occurred while updating config

    See Also
    --------
    load_config

    """

    if config_path.is_file():
        try:
            new_config = OmegaConf.load(config_path)
            cfg = OmegaConf.merge(cfg, new_config)
        except Exception as err:
            raise UpdateConfigError(f"Cannot update config from '{config_path}': {err}") from err
        else:
            if OmegaConf.is_missing(cfg, '__config_paths__'):
                cfg.__config_paths__ = {}
            cfg.__config_paths__[label] = str(config_path)
    elif not_exist_error:
        raise ConfigFileDoesNotExistError(f"'{config_path}' does not exist.")

    return cfg


def load_config(config_paths: ConfigPaths, custom_config_path: Optional[Path] = None) -> DictConfig:
    """Load and return the configuration object

    Parameters
    ----------
    config_paths : config_paths
        Ordered config paths
    custom_config_path : Path, None
        The custom config file path or None

    Returns
    -------
    config : DictConfig
        The configuration object

    Raises
    ------
    LoadConfigError : Cannot load config from config file(s)

    See Also
    --------
    update_config

    """

    try:
        cfg = OmegaConf.load(config_paths.default_path)
    except Exception as err:
        raise LoadConfigError(
            f"Cannot load default configuration from '{config_paths.default_path}': {err}") from err

    cfg.__config_paths__ = {'default': str(config_paths.default_path)}

    try:
        cfg = update_config(cfg, config_paths.site_path, 'site')
        cfg = update_config(cfg, config_paths.user_path, 'user')

        if custom_config_path:
            cfg = update_config(cfg, custom_config_path, 'custom', not_exist_error=True)
    except Exception as err:
        raise LoadConfigError(f'{err}') from err

    return cfg


def show_config_info(cfg: DictConfig):
    """Show config info via logging
    """

    logger.info("The configuration was loaded from config files:")
    for label, path in cfg.__config_paths__.items():
        logger.info('  {path} ({label})', path=path, label=label)


def load_configs(app_config_path: Optional[Path] = None,
                 log_config_path: Optional[Path] = None) -> DictConfig:
    """Load configuration files
    """

    app_config_path = app_config_path or os.environ.get(APP_CONFIG_ENVVAR)
    log_config_path = log_config_path or os.environ.get(LOG_CONFIG_ENVVAR)

    log_cfg = load_config(log_config_paths, custom_config_path=log_config_path)
    configure_logger(log_cfg)

    cfg = load_config(app_config_paths, custom_config_path=app_config_path)
    show_config_info(cfg)

    return cfg


config = load_configs()
