import os

from yaml import load

config_path = 'config.yaml'

with open(config_path, encoding='utf-8') as f:
    cont = f.read()

cf = load(cont)
print(cf)


def get_es_args():
    return cf.get('elasticsearch')


es_args = get_es_args()

a = es_args.get("post")
print(a)
