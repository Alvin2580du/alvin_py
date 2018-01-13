# coding:utf-8
"""
Logger config
Author: alvin
"""
import os
import logging
import logging.config as log_conf
from pyduyp.config.conf import get_log_args

log_dir = get_log_args().get('path')
log_name = get_log_args().get('name')
log_level = get_log_args().get('level')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

log_path = os.path.join(log_dir, log_name)

log_config = {
    'version': 1.0,
    'formatters': {
        'detail': {
            'format': '%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
            'datefmt': "%Y-%m-%d %H:%M:%S"
        },
        'simple': {
            'format': '%(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detail'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 1024 * 1024 * 1024 * 5,
            'backupCount': 10,
            'filename': log_path,
            'level': 'DEBUG',
            'formatter': 'detail',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'crawler': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        'parser': {
            'handlers': ['file'],
            'level': 'INFO',
        },
        'other': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
        'storage': {
            'handlers': ['file'],
            'level': 'DEBUG',
        },
        'log': {
            'handlers': ['console', 'file'],
            'level': log_level,
        },
    }
}

log_conf.dictConfig(log_config)

other = logging.getLogger('other')
storage = logging.getLogger('storage')
log = logging.getLogger('log')

__all__ = ['other', 'storage', 'log']

