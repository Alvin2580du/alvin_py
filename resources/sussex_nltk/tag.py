"""
The tag module provides access to the Stanford and Carnegie Mellon twitter
part-of-speech taggers.

The Stanford tagger has four different models trained on data that has
been preprocessed differently. 

* *wsj-0-18-bidirectional-distsim.tagger* Trained on WSJ sections 0-18 using a
  bidirectional architecture and including word shape and distributional
  similarity features.

  Penn Treebank tagset.

  Performance: 97.28% correct on WSJ 19-21 (90.46% correct on unknown words)

* *wsj-0-18-left3words.tagger* Trained on WSJ sections 0-18 using the left3words
  architecture and includes word shape features.
  
  Penn tagset.
  
  Performance: 96.97% correct on WSJ 19-21 (88.85% correct on unknown words)

* *english-left3words-distsim.tagger* Trained on WSJ sections 0-18 and extra
  parser training data using the left3words architecture and includes word
  shape and distributional similarity features.
  
  Penn tagset.

* *english-bidirectional-distsim.tagger* Trained on WSJ sections 0-18 using a 
  bidirectional architecture and including word shape and distributional
  similarity features.
  
  Penn Treebank tagset.
    
.. codeauthor::
    Matti Lyra
"""

from . import cmu


def twitter_tag_batch(sents):
    '''Tags a batch of sentences using the CMU twitter tokenizer.

    Calling the batch method is faster than sequentially calling
    `twitter_tokenize` is for a large number of sentences.

    :param list sents: list of sentences, each sentence should be a string.
    :return: list of tagged sentences
    :rtype: [[(word, tag), ...], [(word, tag), ...]]
    '''
    _output_data = cmu.tag(sents)
    _output_tokens = []
    sent = []
    for line in _output_data.split('\n'):
        if line.strip() == '':
            _output_tokens.append( sent )
            sent = []
        else:
            token,_,pos_tag = line.partition('\t')
            sent.append( (token.strip(),pos_tag) )
    
    return _output_tokens[:-1]


def twitter_tag(sent):
    """Part-of-speech tag a sentence using the CMU twitter tagger.

    :param str sent: The sentence as a single string.
    :return: tagged sentence
    :rtype: [(word, tag), ...]
    """
    return twitter_tag_batch([sent])
