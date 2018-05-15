import stanfordcorenlp

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.stanford import StanfordDependencyParser

nlp = StanfordCoreNLP(r"D:\jar", lang='zh')
# chi_parser = StanfordDependencyParser(r"D:\jar\stanford-chinese-corenlp-2016-10-31-models.jar")

sentence = '清华大学位于北京。'
print(nlp.word_tokenize(sentence))
print(nlp.pos_tag(sentence))
print(nlp.ner(sentence))
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))
print(list(nlp.parse(sentence.split())))

# res = list(chi_parser.parse(u'四川 已 成为 中国 西部 对外开放 中 升起 的 一 颗 明星'.split()))
# for row in res[0].triples():
#     print(row)
