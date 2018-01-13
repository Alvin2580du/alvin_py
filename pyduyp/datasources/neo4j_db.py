# coding=utf-8

"""
create mysql use sqlalchemy
Author: alvin
"""

from py2neo import Graph
from pyduyp.config.conf import get_neo4j_args
from pyduyp.logger.log import log

args = get_neo4j_args()
uri = "http://{}:{}".format(args.get('host'), args.get('port'))
log.debug("get neo4j url: {}".format(uri))
graph = Graph(uri, username=args.get('user'), password=args.get('pass'))

__all__ = ['graph']

