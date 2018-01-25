
from functools import wraps


def es_get_decorator(func):
    @wraps(func)
    def es_get(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('es get response error, detail {}'.format(e))
    return es_get
