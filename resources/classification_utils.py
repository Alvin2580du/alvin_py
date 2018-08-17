from random import sample
from sussex_nltk.corpus_readers import AmazonReviewCorpusReader
from nltk.probability import FreqDist
from functools import reduce
from nltk.corpus import stopwords
from nltk.classify.api import ClassifierI
from sussex_nltk.stats import evaluate_wordlist_classifier
import random
import collections
import math


def remove_stopwords_and_punctuation(words):
    return [w for w in words if w.isalpha() and w not in stopwords]


def get_all_words(amazon_reviews):
    return reduce(lambda words, review: words + review.words(), amazon_reviews, [])


def most_frequent_words(freqdist, k):
    return [word for word, count in freqdist.most_common(k)]


class SimpleClassifier(ClassifierI):
    def __init__(self, pos, neg):
        self._pos = pos
        self._neg = neg

    def classify(self, words):
        score = 0

        # add code here that assigns an appropriate value to score
        for word in words:
            if word in self._pos:
                score += 1
            if word in self._neg:
                score -= 1
        if score < 0:
            return "N"
        if score > 0:
            return "P"
        return random.choice(["N", "P"])

    def batch_classify(self, docs):
        return [self.classify(doc.words() if hasattr(doc, 'words') else doc) for doc in docs]

    def labels(self):
        return ("P", "N")


def split_data(data, ratio=0.7):
    data = list(data)

    n = len(data)  # Found out number of samples present
    train_indices = sample(range(n), int(n * ratio))  # Randomly select training indices
    test_indices = list(set(range(n)) - set(train_indices))  # Randomly select testing indices

    training_data = [data[i] for i in train_indices]  # Use training indices to select data
    testing_data = [data[i] for i in test_indices]  # Use testing indices to select data

    return (training_data, testing_data)  # Return split data


def format_data(reviews, label, feature_extraction_fn=None):
    if feature_extraction_fn is None:  # If a feature extraction function is not provided, use simply the words of the review as features
        data = [(dict([(feature, True) for feature in review.words()]), label) for review in reviews]
    else:
        data = [(dict([(feature, True) for feature in feature_extraction_fn(review)]), label) for review in reviews]
    return data


def get_train_test_data(corpus_reader, split=0.7):
    pos_train, pos_test = split_data(corpus_reader.positive().documents(), split)
    neg_train, neg_test = split_data(corpus_reader.negative().documents(), split)
    return pos_train, neg_train, pos_test, neg_test


def get_formatted_train_test_data(category, feature_extractor=None, split=0.7):
    '''
    Helper function. Splits data evenly across positive and negative, and then formats it
    ready for naive bayes. You can also optionally pass in your custom feature extractor 
    (see next section), and a custom split ratio.
    '''
    arcr = AmazonReviewCorpusReader()
    pos_train, pos_test = split_data(arcr.positive().category(category).documents(), split)
    neg_train, neg_test = split_data(arcr.negative().category(category).documents(), split)
    train = format_data(pos_train, "pos", feature_extractor) + format_data(neg_train, "neg", feature_extractor)
    test = format_data(pos_test, "pos", feature_extractor) + format_data(neg_test, "neg", feature_extractor)
    return test, train


def class_priors(data):
    doc_counts = collections.defaultdict(int)
    priors = collections.defaultdict(float)
    # first we get the document count for each class
    for doc, c in data:
        doc_counts[c] += 1
    # now we add counts to achieve add-one smoothing
    for c in doc_counts:
        doc_counts[c] += 1
    # now we compute the probabilities 
    # we must add len(doc_counts) to the denominator because of the add-one smoothing
    for c in doc_counts.keys():
        priors[c] = doc_counts[c] / (len(data) + len(doc_counts))
    return priors


def cond_probs(training_data):
    # c_probs will hold our conditional probabilities
    c_probs = collections.defaultdict(lambda: collections.defaultdict(float))
    # docs_with_word is a mapping from a class to a mapping from a word to number of documents of that category the word appeared in 
    docs_with_word = collections.defaultdict(lambda: collections.defaultdict(int))
    # tot_words is a mapping from a class to the total number of words documents of that class
    tot_words = collections.defaultdict(int)

    # first get the counts of words in documents of a class and total word count per class
    for doc, c in training_data:
        for word in doc:
            docs_with_word[c][word] += 1
            tot_words[c] += 1

    # next, add the add-one smoothing counts
    known_vocab = known_vocabulary(training_data)
    for c in docs_with_word.keys():
        for word in known_vocab:
            docs_with_word[c][word] += 1
            # update tot_words to account for the additional (hallucinated) counts
        tot_words[c] += len(known_vocab)

    # now compute the conditional probabilities
    for c in docs_with_word.keys():
        for word in docs_with_word[c].keys():
            c_probs[c][word] = docs_with_word[c][word] / tot_words[c]

    return c_probs


def known_vocabulary(training_data):
    vocab = set()
    for doc, c in training_data:
        for word in doc:
            vocab.add(word)
    return vocab


def classify(doc, priors, c_probs, known_vocab):
    class_scores = collections.defaultdict(lambda: 0)
    for c in priors.keys():
        class_scores[c] += math.log(priors[c])
        for word in doc:
            if word in known_vocab:
                class_scores[c] += math.log(c_probs[c][word])
    best_score = max(class_scores.values())
    return random.choice([c for c in class_scores.keys() if class_scores[c] == best_score])


def NB_evaluate(test_data, priors, c_probs, known_vocab):
    num_correct = 0
    for doc, c in test_data:
        predicted_class = classify(doc, priors, c_probs, known_vocab)
        if predicted_class == c:
            num_correct += 1
    return num_correct / len(test_data)


def run_WL(pos_train, neg_train, pos_test, neg_test, k):
    pos_freqdist = FreqDist(remove_stopwords_and_punctuation(get_all_words(pos_train)))
    neg_freqdist = FreqDist(remove_stopwords_and_punctuation(get_all_words(neg_train)))
    top_pos = most_frequent_words(pos_freqdist, k)
    top_neg = most_frequent_words(neg_freqdist, k)
    classifier = SimpleClassifier(top_pos, top_neg)
    return evaluate_wordlist_classifier(classifier, pos_test, neg_test)


def run_NB(pos_train, neg_train, pos_test, neg_test):
    formatted_pos_train = format_data(pos_train, "pos")
    formatted_neg_train = format_data(neg_train, "neg")
    formatted_train = formatted_pos_train + formatted_neg_train
    formatted_pos_test = format_data(pos_test, "pos")
    formatted_neg_test = format_data(neg_test, "neg")
    formatted_test = formatted_pos_test + formatted_neg_test
    c_priors = class_priors(formatted_train)
    c_probs = cond_probs(formatted_train)
    known_vocab = known_vocabulary(formatted_train)
    return NB_evaluate(formatted_test, c_priors, c_probs, known_vocab)


def run_NB_preformatted(train, test):
    c_priors = class_priors(train)
    c_probs = cond_probs(train)
    known_vocab = known_vocabulary(train)
    return NB_evaluate(test, c_priors, c_probs, known_vocab)


stopwords = stopwords.words('english')


run_NB_preformatted