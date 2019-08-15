from sklearn.datasets import fetch_20newsgroups
from gensim import corpora, models
import en_core_web_sm
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import ortho_group
import numpy as np
from collections import Counter
import pandas as pd

# a small value
EPSILON = 1e-9
PIVOTS_DROPOUT = 0.5
DOC_VECS_DROPOUT = 0.25
DOC_WEIGHTS_INIT = 0.1

# negative sampling power
BETA = 0.75

# i add some noise to the gradient
ETA = 0.4


def preprocess(docs,  min_length, min_counts, max_counts):
    """Tokenize, clean, and encode documents.

    Arguments:
        docs: A list of tuples (index, string), each string is a document.
        nlp: A spaCy object, like nlp = spacy.load('en').
        min_length: An integer, minimum document length.
        min_counts: An integer, minimum count of a word.
        max_counts: An integer, maximum count of a word.

    Returns:
        encoded_docs: A list of tuples (index, list), each list is a document
            with words encoded by integer values.
        decoder: A dict, integer -> word.
        word_counts: A list of integers, counts of words that are in decoder.
            word_counts[i] is the number of occurrences of word decoder[i]
            in all documents in docs.
    """

    tokenized_docs = [(i, doc.split()) for i, doc in tqdm(docs)]
    # remove short documents
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of removed short documents:', n_short_docs)

    # remove some tokens
    counts = _count_unique_tokens(tokenized_docs)
    tokenized_docs = _remove_tokens(tokenized_docs, counts, min_counts, max_counts)
    n_short_docs = sum(1 for i, doc in tokenized_docs if len(doc) < min_length)
    tokenized_docs = [(i, doc) for i, doc in tokenized_docs if len(doc) >= min_length]
    print('number of additionally removed short documents:', n_short_docs)

    counts = _count_unique_tokens(tokenized_docs)
    encoder, decoder, word_counts = _create_token_encoder(counts)

    print('\nminimum word count number:', word_counts[-1])
    print('this number can be less than MIN_COUNTS because of document removal')

    encoded_docs = _encode(tokenized_docs, encoder)
    return encoded_docs, decoder, word_counts


def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    return [(i, [encoder[t] for t in doc]) for i, doc in tokenized_docs]


def _remove_tokens(tokenized_docs, counts, min_counts, max_counts):
    """
    Words with count < min_counts or count > max_counts
    will be removed.
    """
    total_tokens_count = sum(count for token, count in counts.most_common())
    print('total number of tokens:', total_tokens_count)

    unknown_tokens_count = sum(count for token, count in counts.most_common() if count < min_counts or count > max_counts)
    print('number of tokens to be removed:', unknown_tokens_count)

    keep = {}
    for token, count in counts.most_common():
        keep[token] = min_counts <= count <= max_counts

    return [(i, [t for t in doc if keep[t]]) for i, doc in tokenized_docs]


def _create_token_encoder(counts):
    total_tokens_count = sum(count for token, count in counts.most_common())
    print('total number of tokens:', total_tokens_count)

    encoder = {}
    decoder = {}
    word_counts = []
    i = 0

    for token, count in counts.most_common():
        encoder[token] = i
        decoder[i] = token
        word_counts.append(count)
        i += 1

    return encoder, decoder, word_counts


def get_windows(doc, hws=5):
    """
    For each word in a document get a window around it.

    Arguments:
        doc: a list of words.
        hws: an integer, half window size.

    Returns:
        a list of tuples, each tuple looks like this
            (word w, window around w),
            window around w equals to
            [hws words that come before w] + [hws words that come after w],
            size of the window around w is 2*hws.
            Number of the tuples = len(doc).
    """
    length = len(doc)
    assert length > 2 * hws, 'doc is too short!'

    inside = [(w, doc[(i - hws):i] + doc[(i + 1):(i + hws + 1)])
              for i, w in enumerate(doc[hws:-hws], hws)]

    # for words that are near the beginning or
    # the end of a doc tuples are slightly different
    beginning = [(w, doc[:i] + doc[(i + 1):(2 * hws + 1)])
                 for i, w in enumerate(doc[:hws], 0)]

    end = [(w, doc[-(2 * hws + 1):i] + doc[(i + 1):])
           for i, w in enumerate(doc[-hws:], length - hws)]

    return beginning + inside + end


# i believe this helps optimization.
# the idea is taken from here:
# https://arxiv.org/abs/1511.06807
# 'Adding Gradient Noise Improves Learning for Very Deep Networks'

class loss(nn.Module):
    """The main thing to minimize."""

    def __init__(self, topics, word_vectors, unigram_distribution,
                 n_documents, loss_doc_weights, lambda_const=100.0, num_sampled=15):
        """
        Arguments:
            topics: An instance of 'topic_embedding' class.
            word_vectors: A float tensor of shape [vocab_size, embedding_dim].
                A word embedding.
            unigram_distribution: A float tensor of shape [vocab_size]. A distribution
                from which to sample negative words.
            n_documents: An integer, number of documents in dataset.
            loss_doc_weights: A float tensor with shape [n_documents],
                for weighting each document when computing loss
                before taking average over a batch.
            lambda_const: A float number, strength of dirichlet prior.
            num_sampled: An integer, number of negative words to sample.
        """
        super(loss, self).__init__()

        self.topics = topics
        self.n_topics = topics.n_topics
        self.alpha = 1.0 / self.n_topics
        self.lambda_const = lambda_const
        self.weights = loss_doc_weights

        # document distributions (logits) over the topics
        self.doc_weights = nn.Embedding(n_documents, self.n_topics)
        init.normal(self.doc_weights.weight, std=DOC_WEIGHTS_INIT)

        self.neg = negative_sampling_loss(word_vectors, unigram_distribution, num_sampled)

    def forward(self, doc_indices, pivot_words, target_words):
        """
        Arguments:
            doc_indices: A long tensor of shape [batch_size].
            pivot_words: A long tensor of shape [batch_size].
            target_words: A long tensor of shape [batch_size, window_size].
        Returns:
            A pair of losses, their sum is going to be minimized.
        """

        # shape: [batch_size, n_topics]
        doc_weights = self.doc_weights(doc_indices)

        # for reweighting loss
        w = Variable(self.weights[doc_indices.data])
        w /= w.sum()
        w *= w.size(0)

        # shape: [batch_size, embedding_dim]
        doc_vectors = self.topics(doc_weights)

        neg_loss = self.neg(pivot_words, target_words, doc_vectors, w)
        dirichlet_loss = (w * F.log_softmax(doc_weights).sum(1)).mean()
        dirichlet_loss *= self.lambda_const * (1.0 - self.alpha)

        return neg_loss, dirichlet_loss


class AliasMultinomial(object):
    """
    Fast sampling from a multinomial distribution.
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """

    def __init__(self, probs):
        """
        probs: a float tensor with shape [K].
            It represents probabilities of different outcomes.
            There are K outcomes. Probabilities sum to one.
        """

        K = len(probs)
        self.q = torch.zeros(K).cuda()
        self.J = torch.LongTensor([0] * K).cuda()

        # sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = (self.q[large] - 1.0) + self.q[small]

            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.q.clamp(0.0, 1.0)
        self.J.clamp(0, K - 1)

    def draw(self, N):
        """Draw N samples from the distribution."""

        K = self.J.size(0)
        r = torch.LongTensor(np.random.randint(0, K, size=N)).cuda()
        q = self.q.index_select(0, r)
        j = self.J.index_select(0, r)
        b = torch.bernoulli(q)
        oq = r.mul(b.long())
        oj = j.mul((1 - b).long())
        return oq + oj


class negative_sampling_loss(nn.Module):

    def __init__(self, word_vectors, word_distribution, num_sampled=10):
        """
        Arguments:
            word_vectors: A float tensor of shape [vocab_size, embedding_dim].
                A word representation like, for example, word2vec or GloVe.
            word_distribution: A float tensor of shape [vocab_size]. A distribution
                from which to sample negative words.
            num_sampled: An integer, number of negative words to sample.
        """
        super(negative_sampling_loss, self).__init__()

        vocab_size, embedding_dim = word_vectors.size()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data = word_vectors

        # 'AliasMultinomial' is a lot faster than torch.multinomial
        self.multinomial = AliasMultinomial(word_distribution)

        self.num_sampled = num_sampled
        self.embedding_dim = embedding_dim
        self.dropout1 = nn.Dropout(PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(DOC_VECS_DROPOUT)

    def forward(self, pivot_words, target_words, doc_vectors, loss_doc_weights):
        """
        Arguments:
            pivot_words: A long tensor of shape [batch_size].
            target_words: A long tensor of shape [batch_size, window_size].
                Windows around pivot words.
            doc_vectors: A float tensor of shape [batch_size, embedding_dim].
                Documents embeddings.
            loss_doc_weights: A float tensor of shape [batch_size].

        Returns:
            A scalar.
        """

        batch_size, window_size = target_words.size()
        # shape: [batch_size, embedding_dim]
        pivot_vectors = self.embedding(pivot_words)
        # shapes: [batch_size, embedding_dim]
        pivot_vectors = self.dropout1(pivot_vectors)
        doc_vectors = self.dropout2(doc_vectors)
        context_vectors = doc_vectors + pivot_vectors
        # shape: [batch_size, window_size, embedding_dim]
        targets = self.embedding(target_words)
        # shape: [batch_size, 1, embedding_dim]
        unsqueezed_context = context_vectors.unsqueeze(1)
        # compute dot product between a context vector
        # and each word vector in the window,
        # shape: [batch_size, window_size]
        log_targets = (targets * unsqueezed_context).sum(2).sigmoid() \
            .clamp(min=EPSILON).log()

        # sample negative words for each word in the window,
        # shape: [batch_size*window_size*num_sampled]
        noise = self.multinomial.draw(batch_size * window_size * self.num_sampled)
        noise = Variable(noise).view(batch_size, window_size * self.num_sampled)

        # shape: [batch_size, window_size*num_sampled, embedding_dim]
        noise = self.embedding(noise)
        noise = noise.view(batch_size, window_size, self.num_sampled, self.embedding_dim)

        # shape: [batch_size, 1, 1, embedding_dim]
        unsqueezed_context = context_vectors.unsqueeze(1).unsqueeze(1)

        # compute dot product between a context vector
        # and each negative word's vector for each word in the window,
        # then sum over negative words,
        # shape: [batch_size, window_size]
        sum_log_sampled = (noise * unsqueezed_context).sum(3).neg().sigmoid() \
            .clamp(min=EPSILON).log().sum(2)

        neg_loss = log_targets + sum_log_sampled

        # sum over the window, then take mean over the batch
        # shape: []
        return (loss_doc_weights * neg_loss.sum(1)).mean().neg()


class topic_embedding(nn.Module):

    def __init__(self, n_topics, embedding_dim):
        """
        Arguments:
            embedding_dim: An integer.
            n_topics: An integer.
        """
        super(topic_embedding, self).__init__()

        # initialize topic vectors by a random orthogonal matrix
        assert n_topics < embedding_dim
        topic_vectors = ortho_group.rvs(embedding_dim)
        topic_vectors = topic_vectors[0:n_topics]
        topic_vectors = torch.FloatTensor(topic_vectors)

        self.topic_vectors = nn.Parameter(topic_vectors)
        self.n_topics = n_topics

    def forward(self, doc_weights):
        """Embed a batch of documents.

        Arguments:
            doc_weights: A float tensor of shape [batch_size, n_topics],
                document distributions (logits) over the topics.

        Returns:
            A float tensor of shape [batch_size, embedding_dim].
        """

        doc_probs = F.softmax(doc_weights)
        # shape: [batch_size, n_topics, 1]
        unsqueezed_doc_probs = doc_probs.unsqueeze(2)
        # shape: [1, n_topics, embedding_dim]
        unsqueezed_topic_vectors = self.topic_vectors.unsqueeze(0)
        # linear combination of topic vectors weighted by probabilities,
        # shape: [batch_size, embedding_dim]
        doc_vectors = (unsqueezed_doc_probs * unsqueezed_topic_vectors).sum(1)

        return doc_vectors


def train(data, unigram_distribution, word_vectors,
          doc_weights_init=None, n_topics=25,
          batch_size=4096, n_epochs=200,
          lambda_const=100.0, num_sampled=15,
          topics_weight_decay=1e-2,
          topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3,
          save_every=10, grad_clip=5.0):
    """Trains a lda2vec model. Saves the trained model and logs.

    'data' consists of windows around words. Each row in 'data' contains:
    id of a document, id of a word, 'window_size' words around the word.

    Arguments:
        data: A numpy int array with shape [n_windows, window_size + 2].
        unigram_distribution: A numpy float array with shape [vocab_size].
        word_vectors: A numpy float array with shape [vocab_size, embedding_dim].
        doc_weights_init: A numpy float array with shape [n_documents, n_topics] or None.
        n_topics: An integer.
        batch_size: An integer.
        n_epochs: An integer.
        lambda_const: A float number, strength of dirichlet prior.
        num_sampled: An integer, number of negative words to sample.
        topics_weight_decay: A float number, L2 regularization for topic vectors.
        topics_lr: A float number, learning rate for topic vectors.
        doc_weights_lr: A float number, learning rate for document weights.
        word_vecs_lr: A float number, learning rate for word vectors.
        save_every: An integer, save the model from time to time.
        grad_clip: A float number, clip gradients by absolute value.
    """

    n_windows = len(data)
    n_documents = len(np.unique(data[:, 0]))
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
    print('number of documents:', n_documents)
    print('number of windows:', n_windows)
    print('number of topics:', n_topics)
    print('vocabulary size:', vocab_size)
    print('word embedding dim:', embedding_dim)

    # each document has different length,
    # so larger documents will have stronger gradient.
    # to alleviate this problem i reweight loss
    doc_ids = data[:, 0]
    unique_docs, counts = np.unique(doc_ids, return_counts=True)
    weights = np.zeros((len(unique_docs),), 'float32')
    for i, j in enumerate(unique_docs):
        # longer a document -> lower the document weight when computing loss
        weights[j] = 1.0 / np.log(counts[i])
    weights = torch.FloatTensor(weights).cuda()

    # prepare word distribution
    unigram_distribution = torch.FloatTensor(unigram_distribution ** BETA)
    unigram_distribution /= unigram_distribution.sum()
    unigram_distribution = unigram_distribution.cuda()

    # create a data feeder
    dataset = SimpleDataset(torch.LongTensor(data))
    iterator = DataLoader(
        dataset, batch_size=batch_size, num_workers=4,
        shuffle=True, pin_memory=True, drop_last=False
    )

    # create a lda2vec model
    topics = topic_embedding(n_topics, embedding_dim)
    word_vectors = torch.FloatTensor(word_vectors)
    model = loss(topics, word_vectors, unigram_distribution, n_documents, weights, lambda_const, num_sampled)
    model.cuda()

    if doc_weights_init is not None:
        model.doc_weights.weight.data = torch.FloatTensor(doc_weights_init).cuda()

    params = [
        {'params': [model.topics.topic_vectors],
         'lr': topics_lr, 'weight_decay': topics_weight_decay},
        {'params': [model.doc_weights.weight],
         'lr': doc_weights_lr},
        {'params': [model.neg.embedding.weight],
         'lr': word_vecs_lr}
    ]
    optimizer = optim.Adam(params)
    n_batches = math.ceil(n_windows / batch_size)
    print('number of batches:', n_batches, '\n')
    losses = []  # collect all losses here
    doc_weights_shape = model.doc_weights.weight.size()

    model.train()
    try:
        for epoch in range(1, n_epochs + 1):
            print('epoch', epoch)
            running_neg_loss = 0.0
            running_dirichlet_loss = 0.0
            for batch in tqdm(iterator):
                batch = Variable(batch.cuda())
                doc_indices = batch[:, 0]
                pivot_words = batch[:, 1]
                target_words = batch[:, 2:]
                neg_loss, dirichlet_loss = model(doc_indices, pivot_words, target_words)
                total_loss = neg_loss + dirichlet_loss
                optimizer.zero_grad()
                total_loss.backward()
                # level of noise becomes lower as training goes on
                sigma = ETA / epoch ** 0.55
                noise = sigma * Variable(torch.randn(doc_weights_shape).cuda())
                model.doc_weights.weight.grad += noise

                # gradient clipping
                for p in model.parameters():
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

                n_samples = batch.size(0)
                running_neg_loss += neg_loss.data[0] * n_samples
                running_dirichlet_loss += dirichlet_loss.data[0] * n_samples

            losses += [(epoch, running_neg_loss / n_windows, running_dirichlet_loss / n_windows)]
            print('{0:.2f} {1:.2f}'.format(*losses[-1][1:]))
            if epoch % save_every == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_epoch_model_state.pytorch')

    except (KeyboardInterrupt, SystemExit):
        print(' Interruption detected, exiting the program...')

    _write_training_logs(losses)
    torch.save(model.state_dict(), 'model_state.pytorch')


def _write_training_logs(losses):
    with open('training_logs.txt', 'w') as f:
        column_names = 'epoch,negative_sampling_loss,dirichlet_prior_loss\n'
        f.write(column_names)
        for i in losses:
            values = ('{0},{1:.3f},{2:.3f}\n').format(*i)
            f.write(values)


class SimpleDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


if __name__ == '__main__':
    method = 'preprocess'

    if method == 'preprocess':
        MIN_COUNTS = 20
        MAX_COUNTS = 100
        MIN_LENGTH = 15
        HALF_WINDOW_SIZE = 5

        data = pd.read_excel("dataAll.xlsx").head(300)
        train_set = []
        docs = data['content'].values

        # store an index with a document
        docs = [(i, doc) for i, doc in enumerate(docs)]
        encoded_docs, decoder, word_counts = preprocess(docs, MIN_LENGTH, MIN_COUNTS, MAX_COUNTS)
        print(decoder)
        # new ids will be created for the documents.
        # create a way of restoring initial ids:
        doc_decoder = {i: doc_id for i, (doc_id, doc) in enumerate(encoded_docs)}

        data = []
        # new ids are created here
        for index, (_, doc) in tqdm(enumerate(encoded_docs)):
            windows = get_windows(doc, HALF_WINDOW_SIZE)
            # index represents id of a document,
            # windows is a list of (word, window around this word),
            # where word is in the document
            data += [[index, w[0]] + w[1] for w in windows]

        data = np.array(data, dtype='int64')
        print(data.shape)

        word_counts = np.array(word_counts)
        unigram_distribution = word_counts / sum(word_counts)

        vocab_size = len(decoder)
        embedding_dim = 50

        # train a skip-gram word2vec model
        texts = [[str(j) for j in doc] for i, doc in encoded_docs]
        model = models.Word2Vec(texts, size=embedding_dim, window=5, workers=4, sg=1, negative=15, iter=70)
        model.init_sims(replace=True)

        word_vectors = np.zeros((vocab_size, embedding_dim)).astype('float32')
        for i in decoder:
            print(i)
            word_vectors[i] = model.wv[str(i)]

        texts = [[decoder[j] for j in doc] for i, doc in encoded_docs]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        n_topics = 12
        lda_model = models.LdaModel(corpus, alpha=0.9, id2word=dictionary, num_topics=n_topics)
        corpus_lda = lda_model[corpus]

        doc_weights_init = np.zeros((len(corpus_lda), n_topics))
        for i in tqdm(range(len(corpus_lda))):
            topics = corpus_lda[i]
            for j, prob in topics:
                doc_weights_init[i, j] = prob

        np.save('data.npy', data)
        np.save('word_vectors.npy', word_vectors)
        np.save('unigram_distribution.npy', unigram_distribution)
        np.save('decoder.npy', decoder)
        np.save('doc_decoder.npy', doc_decoder)
        np.save('doc_weights_init.npy', doc_weights_init)

    if method == 'train':
        data = np.load('data.npy')
        unigram_distribution = np.load('unigram_distribution.npy')
        word_vectors = np.load('word_vectors.npy')
        doc_weights_init = np.load('doc_weights_init.npy')

        # transform to logits
        doc_weights_init = np.log(doc_weights_init + 1e-4)

        # make distribution softer
        temperature = 7.0
        doc_weights_init /= temperature

        # if you want to train the model like in the original paper set doc_weights_init=None
        # doc_weights_init = None

        n_topics = 12
        batch_size = 4096
        n_epochs = 10
        lambda_const = 100.0
        num_sampled = 15
        topics_weight_decay = 1e-2
        topics_lr = 1e-3
        doc_weights_lr = 1e-3
        word_vecs_lr = 1e-3
        save_every = 10
        grad_clip = 5.0

        n_windows = len(data)
        n_documents = len(np.unique(data[:, 0]))
        embedding_dim = word_vectors.shape[1]
        vocab_size = len(unigram_distribution)
        print('number of documents:', n_documents)
        print('number of windows:', n_windows)
        print('number of topics:', n_topics)
        print('vocabulary size:', vocab_size)
        print('word embedding dim:', embedding_dim)

        # each document has different length,
        # so larger documents will have stronger gradient.
        # to alleviate this problem i reweight loss
        doc_ids = data[:, 0]
        unique_docs, counts = np.unique(doc_ids, return_counts=True)
        weights = np.zeros((len(unique_docs),), 'float32')
        for i, j in enumerate(unique_docs):
            # longer a document -> lower the document weight when computing loss
            weights[j] = 1.0 / np.log(counts[i])
        weights = torch.FloatTensor(weights).cuda()
        # prepare word distribution
        unigram_distribution = torch.FloatTensor(unigram_distribution ** BETA)
        unigram_distribution /= unigram_distribution.sum()
        unigram_distribution = unigram_distribution.cuda()

        # create a data feeder
        dataset = SimpleDataset(torch.LongTensor(data))
        iterator = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)

        # create a lda2vec model
        topics = topic_embedding(n_topics, embedding_dim)
        word_vectors = torch.FloatTensor(word_vectors)
        model = loss(topics, word_vectors, unigram_distribution,
                     n_documents, weights, lambda_const=lambda_const, num_sampled=num_sampled)
        model.cuda()
        print(model, 'model')

        if doc_weights_init is not None:
            model.doc_weights.weight.data = torch.FloatTensor(doc_weights_init).cuda()

        params = [
            {'params': [model.topics.topic_vectors],
             'lr': topics_lr, 'weight_decay': topics_weight_decay},
            {'params': [model.doc_weights.weight],
             'lr': doc_weights_lr},
            {'params': [model.neg.embedding.weight],
             'lr': word_vecs_lr}
        ]
        optimizer = optim.Adam(params)
        n_batches = math.ceil(n_windows / batch_size)
        print('number of batches:', n_batches, '\n')
        losses = []  # collect all losses here
        doc_weights_shape = model.doc_weights.weight.size()
        print("doc_weights_shape", doc_weights_shape)
        model.train()

        for epoch in range(1, n_epochs + 1):
            print('epoch', epoch)
            running_neg_loss = 0.0
            running_dirichlet_loss = 0.0
            for batch in tqdm(iterator):
                batch = Variable(batch.cuda())
                doc_indices = batch[:, 0]
                pivot_words = batch[:, 1]
                target_words = batch[:, 2:]
                neg_loss, dirichlet_loss = model(doc_indices, pivot_words, target_words)
                total_loss = neg_loss + dirichlet_loss
                optimizer.zero_grad()
                total_loss.backward()
                # level of noise becomes lower as training goes on
                sigma = ETA / epoch ** 0.55
                noise = sigma * Variable(torch.randn(doc_weights_shape).cuda())
                model.doc_weights.weight.grad += noise

                # gradient clipping
                for p in model.parameters():
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)

                optimizer.step()

                n_samples = batch.size(0)
                # running_neg_loss += neg_loss.data[0] * n_samples
                running_neg_loss += neg_loss.item() * n_samples
                running_dirichlet_loss += dirichlet_loss.item() * n_samples

            losses += [(epoch, running_neg_loss / n_windows, running_dirichlet_loss / n_windows)]
            print('{0:.2f} {1:.2f}'.format(*losses[-1][1:]))
            if epoch % save_every == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_epoch_model_state.pytorch')

        _write_training_logs(losses)
        torch.save(model.state_dict(), 'model_state.pytorch')

    if method == 'visulize':
        import numpy as np
        from sklearn.datasets import fetch_20newsgroups
        import torch
        import matplotlib.pyplot as plt
        from MulticoreTSNE import MulticoreTSNE as TSNE


        def softmax(x):
            # x has shape [batch_size, n_classes]
            e = np.exp(x)
            n = np.sum(e, 1, keepdims=True)
            return e / n


        dataset = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), categories=['sci.med'])
        docs = dataset['data']

        # store each document with an initial id
        docs = [(i, doc) for i, doc in enumerate(docs)]

        # "integer -> word" decoder
        decoder = np.load('decoder.npy', allow_pickle=True)[()]

        # for restoring document ids, "id used while training -> initial id"
        doc_decoder = np.load('doc_decoder.npy', allow_pickle=True)[()]

        # original document categories
        targets = dataset['target']
        print(targets)
        target_names = dataset['target_names']
        targets = np.array([targets[doc_decoder[i]] for i in range(len(doc_decoder))])
        print(target_names)
        exit(1)

        state = torch.load('model_state.pytorch', map_location=lambda storage, loc: storage)
        n_topics = 12

        doc_weights = state['doc_weights.weight'].cpu().clone().numpy()
        topic_vectors = state['topics.topic_vectors'].cpu().clone().numpy()
        resulted_word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()

        # distribution over the topics for each document
        topic_dist = softmax(doc_weights)

        # vector representation of the documents
        doc_vecs = np.matmul(topic_dist, topic_vectors)

        similarity = np.matmul(topic_vectors, resulted_word_vectors.T)
        most_similar = similarity.argsort(axis=1)[:, -10:]

        for j in range(n_topics):
            topic_words = ' '.join([decoder[i] for i in reversed(most_similar[j])])
            print('topic', j + 1, ':', topic_words)

        tsne = TSNE(perplexity=200, n_jobs=4)
        X = tsne.fit_transform(doc_vecs.astype('float64'))


        def plot(X):
            # X has shape [n_documents, 2]

            plt.figure(figsize=(16, 9), dpi=120);
            cmap = plt.cm.tab20
            number_of_targets = 20

            for i in range(number_of_targets):

                label = target_names[i]
                size = 15.0
                linewidths = 0.5
                edgecolors = 'k'
                color = cmap(i)

                if 'comp' in label:
                    marker = 'x'
                elif 'sport' in label:
                    marker = 's'
                    edgecolors = 'b'
                elif 'politics' in label:
                    marker = 'o'
                    edgecolors = 'g'
                elif 'religion' in label:
                    marker = 'P'
                    size = 17.0
                elif 'sci' in label:
                    marker = 'o'
                    size = 14.0
                    edgecolors = 'k'
                    linewidths = 1.0
                elif 'atheism' in label:
                    marker = 'P'
                    size = 18.0
                    edgecolors = 'r'
                    linewidths = 0.5
                else:
                    marker = 'v'
                    edgecolors = 'm'

                plt.scatter(
                    X[targets == i, 0],
                    X[targets == i, 1],
                    s=size, c=color, marker=marker,
                    linewidths=linewidths, edgecolors=edgecolors,
                    label=label
                )
            leg = plt.legend()
            leg.get_frame().set_alpha(0.3)


        plot(X)  # learned document vectors

        # different colors and markers represent
        # ground truth labels of each document

        # open this image in new tab to see it better
        doc_weights_init = np.load('doc_weights_init.npy')

        tsne = TSNE(perplexity=200, n_jobs=4)
        Y = tsne.fit_transform(doc_weights_init.astype('float64'))

        # to initialize topic assignments for lda2vec algorithm
        # I run normal LDA and used output distributions over topics
        # of each document

        plot(Y)  # distribution over the topics for each document (output of LDA)

        # different colors and markers represent
        # ground truth labels of each document

        # open this image in new tab to see it better

        tsne = TSNE(perplexity=200, n_jobs=4)
        Z = tsne.fit_transform(topic_dist.astype('float64'))

        plot(Z)  # learned distribution over the topics for each document

        # these are topic assignments as on the plot above
        # but these ones are after the training of lda2vec

        # different colors and markers represent
        # ground truth labels of each document

        # open this image in new tab to see it better
        # distribution of nonzero probabilities
        dist = topic_dist.reshape(-1)
        plt.hist(dist[dist > 0.01], bins=40)
        plt.show()

        # distribution of probabilities for some random topic
        plt.hist(topic_dist[:, 10], bins=40)
        plt.show()
