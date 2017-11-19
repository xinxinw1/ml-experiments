import logging as _logging

from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN

_logger = _logging.getLogger('xin-xin')
_logger.setLevel(INFO)
_handler = _logging.StreamHandler()
_handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)

def log(level, msg, *args, **kwargs):
  _logger.log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
  _logger.debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
  _logger.error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
  _logger.fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
  _logger.info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
  _logger.warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
  _logger.warning(msg, *args, **kwargs)
