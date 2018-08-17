"""
The corpus readers module provides access to corpora not included in the
standard NLTK distribution. The corpora have been prepared by the Text Analytics
Group at the University of Sussex (taglaboratory.org).

Corpus Reader API
=================


Accessing statistics
--------------------

.. function:: enumerate()
                 enumerate_sents()

    Return the number of documents or sentences in a corpus.


Accessing the corpus as a `generator`
-------------------------------------

.. function:: raw()

    Return raw (untokenised) documents, word tokenised or sentence tokenised
    documents from the corpus.


Sampling from the corpus
------------------------

.. function:: sample_words(samplesize=100)
              sample_sents(samplesize=100)
              sample_raw_sents(samplesize=100)
              sample_words_by_sents(samplesize=100)
              sample_words_by_documents(samplesize=100)

   All return random samples from the corpus. The difference between these
   sampling methods is the meaning of `samplesize`.

   Each of the `sample_raw_sents`, `sample_sents` and `sample_words_by_sents`
   sample `samplesize` number of **sentences** and return those sentences as
   raw strings, as a list of lists, or as a list of strings respectively.

   The `sample_words` method samples random words
   across the entire corpus and returns them as a list of strings.

   The `sample_words_by_documents` samples random **documents** and returns
   them as a list of strings, i.e. `samplesize` refers to the number of
   **documents** to sample, not words.


Other
------

In addition to these methods, some corpus readers provide extra
functionality such as `AmazonReviewCorpusReader` that allows accessing
specific categories `AmazonReviewCorpusReader.category(cat='dvd')` or reviews
that express a certain sentiment `AmazonReviewCorpusReader.negative()`.
Similarly the `ReutersCorpusReader` allows accessing certain categories
that are not present in other corpora.
"""

import io
import os
import multiprocessing as mp
from itertools import chain
import gzip
import random
import codecs
from string import punctuation 
try:
    import pickle as pickle
except ImportError:
    import pickle

import nltk
from nltk.corpus.reader.api import CorpusReader
from nltk.tokenize import word_tokenize, sent_tokenize

import sussex_nltk as susx
from sussex_nltk import tokenize


def _get_srl_sent(srl_file):
    sent = []
    line = srl_file.readline()
    while line != '\n':
        sent.append( line )
        line = srl_file.readline()
    return ''.join(sent)
    

def _pre_process_file(paths):
    in_path, out_path = paths

    print(in_path, out_path)
    dir = os.path.dirname(out_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    print("%s writing %s..." % (mp.current_process().name, out_path))
    output_file = open(out_path, 'w', encoding='UTF8')
    for review in _reviews_from_file( in_path ):
        #print "pre processing %s" % review.unique_id 
        review._tokenise_segment()
        #print str(review)

        output_file.write(str(review))

    print("closing %s..." % out_path) 
    output_file.close()


def _sample_generator(gen, samplesize):
    sample = []    
    # Fill in the first samplesize elements:
    try:
        for i, _ in enumerate(range(samplesize)):
            sample.append( next(gen) )
    except StopIteration:
        raise ValueError("Sample '%i' larger than population '%i'."
                         % (samplesize, i))
    
    random.shuffle(sample)  # Randomize their positions
    for i, sent in enumerate(gen, start=samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            sample[r] = sent  # at a decreasing rate, replace random items
    
    return sample


def _amazon_xml_reviews_from_file(fileid,count=[0],start=None,end=None):
    file = io.open(fileid, encoding="latin-1") # this line was file = open(fileid, 'rt')
    data = {}
    i = 0
    for line in file:
        try:
        	i += 1
        	line = bytes(line, "latin-1").decode("utf-8") # added line
        except:
        	pass
        	#print(fileid,line)
        	#print(i)

        if line.startswith('</'):
            node = line.rstrip('>\n ').lstrip('</ ' )
            if node == 'review':            
                if start is None or count[0]>=start: 
#                    review = AmazonReview(data)
                    yield data
                data = {} 
                count[0] += 1
                if end is not None and count[0] == end:
                    break
        elif line.startswith('<'):
            node = line.rstrip('>\n ').lstrip('< ')
        elif start is None or count[0]>=start:
            try :
                data[node] += line
            except KeyError:
                data[node] = line
    file.close()
    raise StopIteration()


def _reviews_from_file(fileid,count=[0],start=None,end=None):
    for data in _amazon_xml_reviews_from_file(fileid, count, start, end):
        review = AmazonReview(data)
        yield review
    raise StopIteration()


class CompressedCorpusReader(CorpusReader):
    """A corpus reader for accessing corpora in a gzip format.
    """
    def __init__(self, fileids = r'.*\.gz', data_folder=''):
        _root = os.path.join(susx._sussex_root, data_folder)
        CorpusReader.__init__(self, _root, fileids)        
        self._n = None
        self._n_sents = None
    
    def raw(self, fileids=None):
        """Returns a generator object over the raw documents in the corpus.
        
        The documents are returned as a raw text string in the order they are
        stored in the corpus file.
        
        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a flat generator over the raw documents in the corpus.
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
        doc_count = 0
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                corpus = gzip.open(fileid, 'rt', encoding='UTF8')
                doc = []
                for line in corpus:
                    if line.strip() == '====':
                        doc_count += 1
                        yield ' '.join(doc)
                        doc = []
                    else:
                        doc.append(line.strip())
                corpus.close()
        raise StopIteration()

    def words(self, fileids=None):
        """Returns a generator over the tokens in the corpus.
        
        The generator iterates over all sentences in the corpus in order. The
        order is determined by the order the documents are returned from the
        file system. Document boundaries are not marked in the generator. The
        produced list is a flat list of strings.

        Sentences are tokenised using `nltk.tokenize.word_tokenize`
        
        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a flat generator over the words in the corpus.
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
        for sent in self.sents(fileids):
            for word in sent:
                yield word
    
    def _raw_sents(self, fileids=None):
        fileids = fileids if fileids is not None else self._fileids
        utf8_reader = codecs.getreader('utf8')
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                corpus = gzip.open(fileid, 'rt', encoding='UTF8')
                corpus = utf8_reader(corpus) 
                doc = []
                for line in corpus:
                    if line.strip() == '====':
                        for sent in doc:
                            yield sent
                        doc = []
                    else:
                        line = line.strip()
                        doc.append(line)
                corpus.close()
        raise StopIteration
        
    def sents(self, fileids=None):
        """A generator over the sentences in the corpus.
        
        The generator iterates over all the sentences in the corpus in order
        such that the documents in the corpus are iterated over in an ordered
        sequence. The order is determined by the order the documents are
        returned from the file system. Document boundaries are not marked in
        the generator. The produced list is a list of lists of strings.
        
        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a generator over the sentences in the corpus.
            Each item in the generator is a list of strings.
        :rtype: generator
        """
        for sent in self._raw_sents(fileids):
            yield word_tokenize(sent)
    
    def enumerate_sents(self):
        """Returns the number of sentences in the corpus.

        :rtype: int
        """
        if self._n_sents is not None:
            return self._n_sents
        
        self._n_sents = 0
        for _ in self._raw_sents():
            self._n_sents += 1
        return self._n_sents
    
    def enumerate(self):
        """Returns the number of documents in the corpus.

        :rtype: int
        """
        if self._n is not None:
            return self._n
        
        self._n = 0
        for _ in self.raw():
            self._n += 1
        
        return self._n
    
    def sample_raw_sents(self, samplesize=100):
        """Return a list of sentences sampled from the corpus.
        
        The method selects random sentences (uniformly) from the corpus up to
        *samplesize* and returns those as a list of lists of strings.

        :param int samplesize: The number of sentences to sample.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns: a nested list of lists of strings.
        :rtype: [[str, ..., str], [str, str, ..., str], ...]
        """
        

        if samplesize <= 0:
            raise ValueError('Can not sample %i sentences.' % samplesize)

        if self._n_sents is None:
            self._n_sents = self.enumerate_sents()
        
        samples = sorted(random.sample(range(self._n_sents), samplesize))
        sampled_sents = []
        
        for fileid in self.fileids():
            fileid = self.abspath(fileid)
            if '._' not  in fileid:
                #print(fileid)
                fh = gzip.open(fileid, 'rt', encoding='UTF8')
                data = fh.read()
                data = data.replace('====\n','') 
                lines = data.split('\n')
#            lines = [line for line in lines if line != '']
            
                while len(samples) > 0 and samples[0] < len(lines):
                    sampled_sents.append( lines[samples[0]] )
                    samples = samples[1:]
            
                samples = [s - len(lines) for s in samples]
                fh.close()

        return sampled_sents
    
    def sample_sents(self, samplesize=100):
        """Return a list of word tokenised sentences sampled from the corpus.
        
        The method selects random sentences (uniformly) from the corpus up to
        *samplesize* and returns those as a list of list of strings.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :param int samplesize: The number of sentences to sample.


        :returns: a list of word tokenised sentences randomly sampled from
            the corpus.

        :rtype: [[str, ..., str], [str, str, ..., str], ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i sentences.' % samplesize)
        sents = self.sample_raw_sents(samplesize)
        for i,sent in enumerate(sents):
            sents[i] = word_tokenize(sent)
        
        return sents
    
    def sample_words_by_sents(self, samplesize=100):
        """Return a list of words sampled by sentence from the corpus.
        
        :param int samplesize: number of sentences to sample from the corpus.

        :returns: a list of strings.

        :rtype: [word, word, ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i sentences.' % samplesize)
        result = []
        for sent in self.sample_sents(samplesize):
            result.extend(sent)
        return result

    def sample_words_by_documents(self, samplesize=100):
        """Returns a random sample of words in the corpus.
        
        The sample is generated by selecting *samplesize* documents from the
        corpus and flattening these documents into a list of strings.
        
        :param int samplesize: number of documents to sample from the corpus.
        
        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns: `samplesize` number of randomly sampled documents as a list
            of words

        :rtype: [word, word, ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i documents.' % samplesize)
        sampled_docs = _sample_generator(self.raw(), samplesize)
        result = []
        for doc in sampled_docs:
            for sent in sent_tokenize(doc):
                for word in word_tokenize(sent):
                    result.append(word)
        return result

    def sample_words(self, samplesize=100):
        """Returns a random sample of words in the corpus.
        
        The sample is generated by selecting *samplesize* documents from the
        corpus and flattening these documents into a list of strings.

        :param int samplesize: number of documents to sample from the corpus.
        
        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns: `samplesize` number of randomly sampled words

        :rtype: [word, word, ...]
        """

        if samplesize <= 0:
            raise ValueError('Can not sample %i words.' %samplesize)

        words = self.sample_words_by_documents(samplesize)
        return random.sample(words, samplesize)


class TwitterCorpusReader(CompressedCorpusReader):
    """Tweets about #teamGB collected during the London 2012 olympics.
    
    The corpus spans a roughly 24 hour period between 7th - 8th of August. 
    """
    def __init__(self, fileids = r'.*\.gz'):
        CompressedCorpusReader.__init__(self, fileids, 'data/twitter')
        self._n = None
        self._doc_count = None
        self._n_sents = 962822
    
    def words(self, fileids=None):
        """Returns a generator over the tokens in the corpus.

        The generator iterates over all the sentences in the corpus in order
        such that the documents in the corpus are iterated over in an ordered
        sequence. The order is determined by the order the documents are
        returned from the file system. Document boundaries are not marked in
        the generator. The produced list is a flat list of strings.

        Sentences are tokenised using the Twitter adapated CMU tokeniser in
        `sussex_nltk.tokenize`.

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a generator over the words in the corpus.
        :rtype: generator
        """
        sents = [sent for sent in self._raw_sents(fileids)]
        sents = tokenize.twitter_tokenize_batch(sents)
        for sent in sents:
            for word in sent:
                yield word
            
    
class MedlineCorpusReader(CompressedCorpusReader):
    """Abstracts of medical research papers from the Medline corpus.
    """
    def __init__(self, fileids = r'.*\.gz'):
        CompressedCorpusReader.__init__(self, fileids, 'data/medline')
        self._n_sents = 1714904
        self._n = None
        self._doc_count = None
        
    
class ReutersCorpusReader(CompressedCorpusReader):
    """A subset of the RCV1 corpus providing newswire text.
    
    The categories provided by the reader are ``'finance'`` and ``'sport'``. The
    documents are stored in a raw format, ie. they are not sentence segmented
    or POS tagged.
    
    `Link RCV1 <http://about.reuters.com/researchandstandards/corpus/>`_
    """
    def __init__(self, fileids = r'.*\.gz'):
        CompressedCorpusReader.__init__(self, fileids, 'data/reuters')
        self._n = None
        self._doc_count = None
        self._n_sents = 1113359
    
    def category(self, cat):
        """Returns a new ReutersCorpusReader over the specified category.
        
        :param str cat: should be either `'finance'` or `'sport'`.
        """
        if not cat:
            return self
        return self._reader([cat])
    
    def finance(self):
        """Returns a ReutersCorpusReader restricted to the *finance* category.
        """
        return self._reader(domains=['finance'])
    
    def sport(self):
        """Returns a ReutersCorpusReader restricted to the *sport* category.
        """
        return self._reader(domains=['sport'])
    
    def _reader(self, domains):
        polarised_fileids = []
        for domain in domains:
            for fileid in self._fileids:
                if domain in fileid:
                    polarised_fileids.append(fileid)
        
        reader = ReutersCorpusReader(fileids=polarised_fileids)
        return reader


class WSJCorpusReader(CorpusReader):
    """The WSJCorpusReader provides access to a subsample of the Penn Treebank.
    
    `Link Penn Treebank <http://www.cis.upenn.edu/~treebank/>`_
    """
    def __init__(self, fileids = r'.*\.mrg'):
        _root = os.path.join(susx._sussex_root, 'data/penn_treebank_npbrac_stanforddeps')
        CorpusReader.__init__(self, _root, fileids)
        self._n = None
        self._n_sents = 51520
    
    def raw(self, fileids=None):
        """Returns a generator object over the raw documents in the corpus.
        
        The documents are returned as a raw text string in the order they are
        stored in the corpus file. All markup the documents may contain is
        removed.
        
        :param list fileids: optional list of file ids that can be used to
            filter down the corpus files where the list of strings is generated
            from.
        """
        fileids = fileids if fileids is not None else self._fileids
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                with open(fileid, 'rt', encoding='UTF8') as doc:
                    doc_text = []
                    for i,line in enumerate(doc):
                        if len(line.strip()) != 0:
                            token = line.split('\t')[1]
                            if i > 0 and token not in punctuation:
                                token = ' %s'%token
                            doc_text.append(token)
                        elif len(doc_text) > 0:
                            yield ''.join(doc_text)
                            doc_text = []
                        
        raise StopIteration()
    
    def words(self, fileids=None):
        """Returns a flat list of the words in the corpus.

        :returns: a generator over the words in the corpus.
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                with open(fileid, 'rt', encoding='UTF8') as doc:
                    for line in doc:
                        if len(line.strip()) != 0:
                            word = line.split('\t')[1]
                            yield word
        return
    
    def tagged_words(self, fileids=None):
        """Returns a flat list of tagged words of the corpus.
        
        :returns: a generator over the words in the corpus. Each item in the
            generator is a tuple consisting of `(token, tag)`
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
        _words = []
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                with open(fileid, 'rt', encoding='UTF8') as doc:
                    for line in doc:
                        if len(line.strip()) != 0:
                            _,token,pos_tag,_,_ = tuple(line.split('\t'))
                            yield (token, pos_tag)
        return
    
    def sents(self, fileids=None):
        """A generator over the sentences in the corpus.
        
        The generator iterates over all the sentences in the corpus in order
        such that the documents in the corpus are iterated over in an ordered
        sequence. The order is determined by the order the documents are
        returned from the file system. Document boundaries are not marked in
        the generator. The produced list is a list of lists of strings.

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a generator over the sentences in the corpus.
            Each item in the generator is a list of strings.
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
        _sents = []
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                with open(fileid, 'r', encoding='UTF8') as doc:
                    sent = []
                    for line in doc:
                        if len(line.strip()) != 0:
                            sent.append('%s'%line.split('\t')[1])
                        else:
                            yield sent
        return

    def tagged_sents(self, fileids=None):
        """A generator over tagged sentences.
        
        The generator iterates over all the sentences in the corpus in order
        such that the documents in the corpus are iterated over in an ordered
        sequence. The order is determined by the order the documents are
        returned from the file system. Document boundaries are not marked in
        the generator. The produced list is a list of lists of strings.

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a flat generator over the sentences in the corpus.
            Each item in the generator is a list of tuples (token, postag).
        :rtype: generator
        """
        fileids = fileids if fileids is not None else self._fileids
#        _sents = []
        for fileid in self.abspaths(fileids):
            if '._' not  in fileid:
                with open(fileid, 'rt', encoding='UTF8') as doc:
                    sent = []
                    for line in doc:
                        if line.strip():
                            parts = line.split('\t')
                            sent.append((parts[1], parts[2]))
                        elif sent:
                            yield sent
                            sent = []
        return
    
    def enumerate_sents(self):
        """Returns the number of sentences in the corpus.

        :rtype: int
        """
        if self._n_sents is not None:
            return self._n_sents
            
        self._n_sents = len([_ for _ in self.sents()])
        return self._n_sents
    
    def enumerate(self):
        """Returns the number of documents in the corpus.

        :rtype: int
        """
        if self._n is not None:
            return self._n
            
        self._n = len(self.fileids())
        return self._n
    
    def sample_sents(self, samplesize=100):
        """Return a list of word tokenised sentences sampled from the corpus.

        The method selects random sentences (uniformly) from the corpus up to
        *samplesize* and returns those as a list of list of strings.

        :param int samplesize: The number of sentences to sample.
        :returns:
            a list of word tokenised sentences randomly sampled from the corpus.
        :rtype: [[str, ..., str], [str, str, ..., str], ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i sentences.' % samplesize)

        sents = [sent for sent in self.sents()]
        sampled_sents = random.sample(sents, samplesize)
        return sampled_sents
    
    def sample_words_by_sents(self, samplesize=100):
        """Return a list of words sampled by sentence from the corpus.

        :param int samplesize: number of sentences to sample from the corpus.

        :returns: a list of strings.
        :rtype: [word, word, ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i words.' % samplesize)
        result = []
        for sent in self.sample_sents(samplesize):
            result.extend(sent)
        return result
    
    def sample_words_by_documents(self, samplesize=100):
        """Returns a random sample of words in the corpus.

        The sample is generated by selecting *samplesize* documents from the
        corpus and flattening these documents into a list of strings.

        :param int samplesize: number of documents to sample from the corpus.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns:
            `samplesize` number of randomly sampled documents as a list of words
        :rtype: [word, word, ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i documents.' % samplesize)
        sampled_docs = _sample_generator(self.raw(), samplesize)
        result = []
        for doc in sampled_docs:
            for sent in sent_tokenize(doc):
                for word in word_tokenize(sent):
                    result.append(word)
        return result
    
    def sample_words(self, samplesize=100):
        """Returns a random sample of words in the corpus.

        The sample is generated by selecting *samplesize* documents from the
        corpus and flattening these documents into a list of strings.

        :param int samplesize: number of documents to sample from the corpus.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns: `samplesize` number of randomly sampled words
        :rtype: [word, word, ...]
        """
        if samplesize <= 0:
            raise ValueError('Can not sample %i documents.' % samplesize)
        words = self.sample_words_by_documents(samplesize)
        return random.sample(words, samplesize)


class AmazonReviewCorpusReader(CorpusReader):
    """The reader provides access to user written product reviews on amazon.com.
    
    The corpus is categorised into ``'dvd','book','kitchen'`` and ``'electronics'``
    and each category is further divided into three sentiment classes
    ``'positive','negative'`` and ``'neutral'``.
    
    Each category contains 1000 reviews for the ``'positive'`` and ``'negative'``
    sentiment classes.   
    """
    def __init__(self, fileids = r'.*\.review'):
        _root = os.path.join(susx._sussex_root, 'data/amazon_customer_reviews')
        self._n_sents = 140443
        CorpusReader.__init__(self, _root, fileids)
        self._n = None

    def category(self, cat):
        """Returns a new AmazonReviewCorpusReader over the specified category.
        
        :param str cat: should be one of ``'kitchen', 'dvd', 'book',
            'electronics'``.
        """
        if not cat:
            return self
        return self._reviews([cat], ['negative', 'positive', 'unlabeled'])
    
    def negative(self, domains = ['books', 'dvd', 'electronics', 'kitchen']):
        """Returns a new AmazonReviewCorpusReader over the negative reviews.
        
        :param list domains: a list of categories.
        """
        return self._reviews(domains, ['negative'])
                    
    def positive(self, domains = ['books', 'dvd', 'electronics', 'kitchen']):
        """Returns a new AmazonReviewCorpusReader over the positive reviews.
        
        :param list domains: a list of categories.
        """
        return self._reviews(domains, ['positive'])

    def unlabeled(self, domains = ['books', 'dvd', 'electronics', 'kitchen']):
        """Returns a new AmazonReviewCorpusReader over the unlabeled reviews.
        
        :param list domains: a list of categories.
        """
        return self._reviews(domains, ['unlabeled'])        
    
    def _reviews(self, domains, polarities):
        polarised_fileids = []
        for domain in domains:
            for polarity in polarities:
                for fileid in self._fileids:
                    if fileid.startswith(domain) and fileid.endswith(polarity+".review"):
                        polarised_fileids.append(fileid)
                        #return self.reviews(self.abspath(fileid))
        
        return AmazonReviewCorpusReader(polarised_fileids)
    
    def words(self):
        """Returns a generator over the tokens in the corpus.

        The generator iterates over all sentences in the corpus in order. The
        order is determined by the order the documents are returned from the
        file system. Document boundaries are not marked in the generator. The
        produced list is a flat list of strings.

        Sentences are tokenised using `nltk.tokenize.word_tokenize`

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a flat generator over the words in the corpus.
        :rtype: generator
        """
        for fileid in self.abspaths(self._fileids):
            if "._" not in fileid:
                for review in _reviews_from_file(fileid):
                    for word in review.words():
                        yield word
        return
    
    def _raw_sents(self):
        for fileid in self.abspaths(self._fileids):
            if "._" not in fileid:
                for review in _reviews_from_file(fileid):
                    for sent in review._data['review_text_tokenised_segmented'].split('\n'):
                        yield sent
    
    def sents(self):
        """A generator over the sentences in the corpus.

        The generator iterates over all the sentences in the corpus in order
        such that the documents in the corpus are iterated over in an ordered
        sequence. The order is determined by the order the documents are
        returned from the file system. Document boundaries are not marked in
        the generator. The produced list is a list of lists of strings.

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a generator over the sentences in the corpus.
            Each item in the generator is a list of strings.
        :rtype: generator
        """
        for fileid in self.abspaths(self._fileids):
            if "._" not in fileid:
                for review in _reviews_from_file(fileid):
                    for sent in review.sents():
                        yield sent 
    
    def raw(self):
        """Returns a generator object over the raw documents in the corpus.

        The documents are returned as a raw text string in the order they are
        stored in the corpus file.

        :param list fileids: Optional list of file ids that can be used to
            filter down the corpus files where the list of strings is
            generated from.

        :returns: a flat generator over the raw documents in the corpus.
        :rtype: generator
        """
        for fileid in self.abspaths(self._fileids):
            if "._" not in fileid:
                for review in _reviews_from_file(fileid):
                    yield review.raw()
        return

    def _pre_process_corpus(self, output_dir, replace_self=False):
        n_cpu = mp.cpu_count()
        print("%d cores detected, using all of them." % (n_cpu))

        #construct arguments in tuples for mapped functions
        inputs = [(self.abspath(fileid), os.path.join(output_dir, fileid)) for fileid in self._fileids]
        
        pool = mp.Pool(processes=n_cpu)
        pool.map(_pre_process_file, inputs, chunksize=n_cpu)
        pool.close()
        pool.join()

        print('pre-processing complete')
        if replace_self:
            tmp = AmazonReviewCorpusReader(output_dir, self._fileids)
            self.__dict__ = tmp.__dict__
                
    def _attach_srl_data(self, srl_path, output_dir, replace_self=False):
        srl_file = open(srl_path, 'rt', encoding='UTF8')
        for fileid in self._fileids:
            if "._" not in fileid:
                path = os.path.join(output_dir, fileid)
                dir = os.path.dirname(path)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                
            print("writing %s..." % path)
            output_file = open(path, 'w', encoding='UTF8')
            
            for review in _reviews_from_file( self.abspath(fileid) ):
                
                print("pre processing %s: %s" % (path, review.unique_id)) 
                #print review.n_sents
                
                #review.srl = ''
                #for i in range(review.n_sents):
                #    review.srl += get_srl_sent(srl_file)
                review.srl = "\n".join( [_get_srl_sent(srl_file) for i in range(review.n_sents)] )
                
                #print review.srl
                output_file.write( str( review ) )
            
            print("closing %s..." % path)    
            output_file.close()
        srl_file.close()
        print('pre-processing complete')
        if replace_self:
            tmp = AmazonReviewCorpusReader(output_dir, self._fileids)
            self.__dict__ = tmp.__dict__
    
    def enumerate_sents(self):
        """Returns the number of sentences in the corpus.

        :rtype: int
        """
        if self._n_sents is not None:
            return self._n_sents
            
        self._n_sents = 0
        for _ in self._raw_sents():
            self._n_sents += 1

        return self._n_sents
       
    def enumerate(self):
        """Returns the number of documents in the corpus.

        :rtype: int
        """
        if self._n is not None:
            return self._n
            
        self._n = 0
        for _ in self.documents():
            self._n += 1

        return self._n    
    
    def _raw_documents(self, start=None, end=None):
        count = [0]
        for fileid in self._fileids:
            if end is not None and count[0] >= end :
                break
            for review in _reviews_from_file(self.abspath(fileid), count, start, end):
                yield review.raw()
    
    def documents(self,start=None,end=None):
        """Generator over the documents in the corpus.
        
        :returns: `AmazonReview` objects.
        :rtype: generator
        """
        count = [0]
        for fileid in self._fileids:
            if "._" not in fileid:
                if not count[0] % 1000 and count[0]:
                    #print "[%d]" % (count[0])
                    pass
                if end is not None and count[0] >= end :
                    break
                for review in _reviews_from_file(self.abspath(fileid), count, start, end):
                    yield review
    
    def sample_raw_sents(self, samplesize=100):
        """Return a list of sentences sampled from the corpus.

        The method selects random sentences (uniformly) from the corpus up to
        *samplesize* and returns those as a list of list of strings.

        :param int samplesize: The number of sentences to sample.
        :returns:
            a list of word tokenised sentences randomly sampled from the corpus.
        :rtype: [str, ..., str]
        """
        sents = [sent for sent in self._raw_sents()]
        sampled_sents = random.sample(sents, samplesize)
        return sampled_sents
    
    def sample_sents(self, samplesize=100):
        """Return a list of word tokenised sentences sampled from the corpus.

        The method selects random sentences (uniformly) from the corpus up to
        *samplesize* and returns those as a list of list of strings.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :param int samplesize: The number of sentences to sample.
        :returns:
            a list of word tokenised sentences randomly sampled from the corpus.
        :rtype: [[str, ..., str], [str, str, ..., str], ...]
        """
        return [word_tokenize(sent) for sent in self.sample_raw_sents(samplesize)]
    
    def sample_words_by_sents(self, samplesize=100):
        """Return a list of words sampled by sentence from the corpus.

        :param int samplesize: number of sentences to sample from the corpus.

        :returns: a list of strings.
        :rtype: [word, word, ...]
        """
        sampled_sents = self.sample_sents(samplesize)
        result = []
        for sent in sampled_sents:
            result.extend(sent)
        return result
    
    def sample_words(self, samplesize=100):
        """Returns a random sample of words in the corpus.

        The sample is generated by selecting *samplesize* documents from the
        corpus and flattening these documents into a list of strings.

        :param int samplesize: number of documents to sample from the corpus.

        :raises: `ValueError` if `samplesize` is not a positive number or is
            larger than the population size.

        :returns: `samplesize` number of randomly sampled words
        :rtype: [word, word, ...]
        """
        sampled_docs = _sample_generator(self.raw(), samplesize)
        result = []
        for doc in sampled_docs:
            for sent in sent_tokenize(doc):
                for word in word_tokenize(sent):
                    result.append(word)
        return random.sample(result, samplesize)


class AmazonReview(object):
    
    def __init__(self, data):
        """ Initialise a single Amazon Review from `data`.

        :param dict data: Dictionary containing the required fields for the
            review object.
        :return: self
        """
        self._sents = {}
        self._data = {}
        self._data['unique_id'] = data['unique_id']
        self._data['asin'] = data['asin']
        self._data['product_name'] = data['product_name']
        self._data['product_type'] = data['product_type']
        self._data['helpful'] = data['helpful']
        self._data['rating'] = float(data['rating'])
        self._data['title'] = data['title']
        self._data['date'] = data['date']
        self._data['reviewer'] = data['reviewer']
        self._data['reviewer_location'] = data['reviewer_location']
        self._data['review_text'] = data['review_text']
        self._data['review_text_tokenised_segmented'] = '' if 'review_text_tokenised_segmented' not in data else data['review_text_tokenised_segmented']
        self._data['srl'] = '' if 'srl' not in data else data['srl']
        self._data['n_sents'] = '' if 'n_sents' not in data else int(data['n_sents'])
    
    def __str__(self):
        """
        output format is sensitive to newlines and whitespace.
        strings have newlines included, numbers do not.
        """
        str = """<review>
<unique_id>
%(unique_id)s</unique_id>
<asin>
%(asin)s</asin>
<product_name>
%(product_name)s</product_name>
<product_type>
%(product_type)s</product_type>
<helpful>
%(helpful)s</helpful>
<rating>
%(rating)s
</rating>
<title>
%(title)s</title>
<date>
%(date)s</date>
<reviewer>
%(reviewer)s</reviewer>
<reviewer_location>
%(reviewer_location)s</reviewer_location>
<review_text>
%(review_text)s</review_text>
<review_text_tokenised_segmented>
%(review_text_tokenised_segmented)s
</review_text_tokenised_segmented>
<n_sents>
%(n_sents)s
</n_sents>
<srl>
%(srl)s</srl>
</review>
""" % (self._data)
        return str
    
    def rating(self):
        """The rating of the review on a scale from 1 to 5.

        :return float:
        """
        return self._data['rating']
    
    def _format_sentences_string(self, word_limit = 70):
        if not self._sents:
            self._tokenise_segment(word_limit)

        one_sent_per_line = "\n".join((" ".join(sent) for sent in self._sents))
        if one_sent_per_line:
            #print sent_per_line
            self._data['n_sents'] = one_sent_per_line.count('\n') + 1
        else:
            self._data['n_sents'] = 0
        #print self.n_sents
        self._data['review_text_tokenised_segmented'] = one_sent_per_line

    def tagged_sents(self):
        """A list of tagged sentences.

        :returns: a list of sentences in the review.
            Each item in the list is a list of tuples `(token, postag)`.
        :rtype: [[(token, pos), ...]]
        """
        srl = self._data['srl']
        
        sents = srl.split("\n\n")
        tagged_sents = []
        for sent in sents:
            
            tokens = sent.split("\n")
            
            tagged_sent = [(token.split("\t")[1], token.split("\t")[4]) for token in tokens if token]
        
            tagged_sents.append(tagged_sent)
        
        return tagged_sents

    def tagged_words(self):
        """A list of tagged words.

        :returns: a list of words in the review.
            Each item in the list is a tuple `(token, postag)`.
        :rtype: [(token, pos), ...]
        """
        sents = self.tagged_sents()
        return [w for w in chain.from_iterable(sents)]
    
    def _tokenise_segment(self, word_limit = 0):
        if self._data['review_text_tokenised_segmented'] != '':
            self._sents = [nltk.word_tokenize(sent) for sent in self._data['review_text_tokenised_segmented'].split('\n') if (not word_limit or len(sent) <= word_limit)]
        else:  
            self._sents = [sent for sent in (nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self._data['review_text'])) if not word_limit or len(sent) <= word_limit]

    def raw(self):
        """Return the raw text of the review.

        :return str:
        """
        return self._data['review_text']
    
    def words(self):
        """Return the review as a flat list of words

        :return list: [word, word, ...]
        """
        if not self._sents:
            self._tokenise_segment()
        words = [word for word in chain(*self._sents)]
        return words
    
    def sents(self):
        """Return the review as a nested list of sentences.

        :return list:
        """
        if not self._sents:
            self._tokenise_segment()
        return self._sents 
