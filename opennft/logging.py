# -*- coding: utf-8 -*-

import sys

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .errors import OpenNftError


STD_CHANNELS = {
    'sys.stdout': sys.stdout,
    'sys.stderr': sys.stderr,
}


class ConfigureLoggingError(OpenNftError):
    pass


def configure_logger(logging_config: DictConfig):
    dict_config = OmegaConf.to_container(logging_config.logging, resolve=True)

    for handler in dict_config.get('handlers', []):
        sink = handler.get('sink')
        if sink:
            handler['sink'] = STD_CHANNELS.get(sink, sink)

    try:
        logger.configure(**dict_config)
    except Exception as err:
        raise ConfigureLoggingError(f'Cannot configure logging: {err}') from err

    for logger_name, enabled in logging_config.loggers.items():
        if enabled:
            logger.enable(logger_name)
        else:
            logger.disable(logger_name)
