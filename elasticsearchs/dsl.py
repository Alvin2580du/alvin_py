body_1 = {
    "query": {
        "bool": {
            "should": [
                {"match": {"title": "War and Peace"}},
                {"match": {"author": "Leo Tolstoy"}},
                {"bool": {
                    "should": [
                        {"match": {"translator": "Constance Garnett"}},
                        {"match": {"translator": "Louise Maude"}}
                    ]
                }}
            ]
        }
    }
}

body_2 = {
    "query": {
        "bool": {
            "should": [
                {"match": {"title": "War and Peace"}},
                {"match": {"author": "Leo Tolstoy"}}
            ]
        }
    }
}

body_3 = {
    "query": {
        "bool": {
            "should": [
                {"match": {
                    "title": {
                        "query": "War and Peace",
                        "boost": 2
                    }}},
                {"match": {
                    "author": {
                        "query": "Leo Tolstoy",
                        "boost": 2
                    }}},
                {"bool": {
                    "should": [
                        {"match": {"translator": "Constance Garnett"}},
                        {"match": {"translator": "Louise Maude"}}
                    ]
                }}
            ]
        }
    }
}
body_4 = {
    "query": {
        "dis_max": {
            "queries": [
                {"match": {"title": "Quick pets"}},
                {"match": {"body": "Quick pets"}}
            ]
        }
    }
}
body_5 = {
    "query": {
        "dis_max": {
            "queries": [
                {"match": {"title": "Quick pets"}},
                {"match": {"body": "Quick pets"}}
            ],
            "tie_breaker": 0.3
        }
    }
}

settings = {
    "settings": {"number_of_shards": 1},
    "mappings": {
        "my_type": {
            "properties": {
                "title": {
                    "type": "string",
                    "analyzer": "ik_smart",
                    "fields": {
                        "std": {
                            "type": "string",
                            "analyzer": "standard"
                        }
                    }
                }
            }
        }
    }
}

body_6 = {
    "query": {
        "multi_match": {
            "query": "Poland Street W1V",
            "type": "most_fields",
            "fields": ["street", "city", "country", "postcode"]
        }
    }
}

body_7 = {
    "query": {
        "bool": {
            "should": [
                {"match": {"street": "Poland Street W1V"}},
                {"match": {"city": "Poland Street W1V"}},
                {"match": {"country": "Poland Street W1V"}},
                {"match": {"postcode": "Poland Street W1V"}}
            ]
        }
    }
}
# slop 参数告诉 match_phrase 查询词条相隔多远时仍然能将文档视为匹配 。 相隔多远的意思是为了让查询和文档匹配你需要移动词条多少次
body_8 = {
    "query": {
        "match_phrase": {
            "title": {
                "query": "quick fox",
                "slop": 1
            }
        }
    }
}
# 正则表达式查询
body_9 = {
    "query": {
        "regexp": {
            "postcode": "W[0-9].+"}
    }
}

body_10 = {
    "match_phrase_prefix": {
        "brand": {
            "query": "walker johnnie bl",
            "slop": 10
        }
    }
}

body_11 = {
    "match_phrase_prefix": {
        "brand": {
            "query": "johnnie walker bl",
            "max_expansions": 50
        }
    }
}

settings_1 = {
    "settings": {
        "analysis": {
            "filter": {
                "trigrams_filter": {
                    "type": "ngram",
                    "min_gram": 3,
                    "max_gram": 3
                }
            },
            "analyzer": {
                "trigrams": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "trigrams_filter"
                    ]
                }
            }
        }
    },
    "mappings": {
        "my_type": {
            "properties": {
                "text": {
                    "type": "string",
                    "analyzer": "trigrams"
                }
            }
        }
    }
}

body_12 = {
    "query": {
        "match": {
            "text": {
                "query": "Gesundheit",
                "minimum_should_match": "80%"  # 至少匹配80%
            }
        }
    }
}

body_13 = {
    "query": {
        "boosting": {
            "positive": {
                "match": {
                    "text": "apple"
                }
            },
            "negative": {
                "match": {
                    "text": "pie tart fruit crumble tree"
                }
            },
            "negative_boost": 0.5
        }
    }
}
# 不考虑tf-idf,之考虑是否出现过
body_14 = {
    "query": {
        "bool": {
            "should": [
                {"constant_score": {"query": {"match": {"description": "{}"}}}},
                {"constant_score": {"query": {"match": {"description": "{}"}}}},
                {"constant_score": {"query": {"match": {"description": "{}"}}}}
            ]
        }
    }
}

