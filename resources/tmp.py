import nltk
from nltk.corpus import gutenberg
print(dir(gutenberg))
exit(1)
import en_core_web_sm

nlp = en_core_web_sm.load()
raw = gutenberg.raw("burgess-busterbrown.txt")

def fun1():
    reader = gcr.GutenbergCorpusReader()
    print(reader)
    authors = reader.get_authors()

    print('The number of authors mentioned in the Gutenberg corpus are:', format(len(authors)))
    for author in authors:
        print("{0}: {1}".format(author, len(authors[author])))

    authors = reader.get_authors()
    tot = 0
    for author in authors:
        tot += len(authors[author])
    works = reader.get_authors_works('De Mille, James')
    for work in works:
        print(work["title"])
    parsed_novel = nlp(works[7]["text"])
    return parsed_novel

fun1()