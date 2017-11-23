import logging as _logging

from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN

_logger = _logging.getLogger('xin-xin')
_logger.setLevel(INFO)
_handler = _logging.StreamHandler()
_handler.setFormatter(_logging.Formatter('[%(levelname)s] %(asctime)-15s [%(filename)s:%(lineno)d] %(message)s', None))
_logger.addHandler(_handler)

log = _logger.log
debug = _logger.debug
error = _logger.error
fatal = _logger.fatal
info = _logger.info
warn = _logger.warn
warning = _logger.warning
