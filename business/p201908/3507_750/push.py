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

EPSILON = 1e-9
PIVOTS_DROPOUT = 0.5
DOC_VECS_DROPOUT = 0.25
DOC_WEIGHTS_INIT = 0.1

BETA = 0.75
ETA = 0.4


def _count_unique_tokens(tokenized_docs):
    tokens = []
    for i, doc in tokenized_docs:
        tokens += doc
    return Counter(tokens)


def _encode(tokenized_docs, encoder):
    return [(i, [encoder[t] for t in doc]) for i, doc in tokenized_docs]


def _remove_tokens(tokenized_docs, counts, min_counts, max_counts):
    total_tokens_count = sum(count for token, count in counts.most_common())
    print('total number of tokens:', total_tokens_count)

    unknown_tokens_count = sum(
        count for token, count in counts.most_common() if count < min_counts or count > max_counts)
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
    length = len(doc)
    assert length > 2 * hws, 'doc is too short!'

    inside = [(w, doc[(i - hws):i] + doc[(i + 1):(i + hws + 1)])
              for i, w in enumerate(doc[hws:-hws], hws)]

    beginning = [(w, doc[:i] + doc[(i + 1):(2 * hws + 1)])
                 for i, w in enumerate(doc[:hws], 0)]

    end = [(w, doc[-(2 * hws + 1):i] + doc[(i + 1):])
           for i, w in enumerate(doc[-hws:], length - hws)]

    return beginning + inside + end


class loss(nn.Module):
    def __init__(self, topics, word_vectors, unigram_distribution,
                 n_documents, loss_doc_weights, lambda_const=100.0, num_sampled=15):
        super(loss, self).__init__()
        self.topics = topics
        self.n_topics = topics.n_topics
        self.alpha = 1.0 / self.n_topics
        self.lambda_const = lambda_const
        self.weights = loss_doc_weights

        self.doc_weights = nn.Embedding(n_documents, self.n_topics)
        init.normal(self.doc_weights.weight, std=DOC_WEIGHTS_INIT)

        self.neg = negative_sampling_loss(word_vectors, unigram_distribution, num_sampled)

    def forward(self, doc_indices, pivot_words, target_words):
        doc_weights = self.doc_weights(doc_indices)

        w = Variable(self.weights[doc_indices.data])
        w /= w.sum()
        w *= w.size(0)
        doc_vectors = self.topics(doc_weights)
        neg_loss = self.neg(pivot_words, target_words, doc_vectors, w)
        dirichlet_loss = (w * F.log_softmax(doc_weights).sum(1)).mean()
        dirichlet_loss *= self.lambda_const * (1.0 - self.alpha)

        return neg_loss, dirichlet_loss


class AliasMultinomial(object):
    def __init__(self, probs):

        K = len(probs)
        self.q = torch.zeros(K).cuda()
        self.J = torch.LongTensor([0] * K).cuda()

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

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
        super(negative_sampling_loss, self).__init__()

        vocab_size, embedding_dim = word_vectors.size()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data = word_vectors
        self.multinomial = AliasMultinomial(word_distribution)
        self.num_sampled = num_sampled
        self.embedding_dim = embedding_dim
        self.dropout1 = nn.Dropout(PIVOTS_DROPOUT)
        self.dropout2 = nn.Dropout(DOC_VECS_DROPOUT)

    def forward(self, pivot_words, target_words, doc_vectors, loss_doc_weights):
        batch_size, window_size = target_words.size()
        pivot_vectors = self.embedding(pivot_words)
        pivot_vectors = self.dropout1(pivot_vectors)
        doc_vectors = self.dropout2(doc_vectors)
        context_vectors = doc_vectors + pivot_vectors
        targets = self.embedding(target_words)
        unsqueezed_context = context_vectors.unsqueeze(1)

        log_targets = (targets * unsqueezed_context).sum(2).sigmoid() \
            .clamp(min=EPSILON).log()

        noise = self.multinomial.draw(batch_size * window_size * self.num_sampled)
        noise = Variable(noise).view(batch_size, window_size * self.num_sampled)

        noise = self.embedding(noise)
        noise = noise.view(batch_size, window_size, self.num_sampled, self.embedding_dim)

        unsqueezed_context = context_vectors.unsqueeze(1).unsqueeze(1)

        sum_log_sampled = (noise * unsqueezed_context).sum(3).neg().sigmoid() \
            .clamp(min=EPSILON).log().sum(2)
        neg_loss = log_targets + sum_log_sampled
        return (loss_doc_weights * neg_loss.sum(1)).mean().neg()


class topic_embedding(nn.Module):

    def __init__(self, n_topics, embedding_dim):
        super(topic_embedding, self).__init__()

        assert n_topics < embedding_dim
        topic_vectors = ortho_group.rvs(embedding_dim)
        topic_vectors = topic_vectors[0:n_topics]
        topic_vectors = torch.FloatTensor(topic_vectors)

        self.topic_vectors = nn.Parameter(topic_vectors)
        self.n_topics = n_topics

    def forward(self, doc_weights):
        doc_probs = F.softmax(doc_weights)
        unsqueezed_doc_probs = doc_probs.unsqueeze(2)
        unsqueezed_topic_vectors = self.topic_vectors.unsqueeze(0)
        doc_vectors = (unsqueezed_doc_probs * unsqueezed_topic_vectors).sum(1)

        return doc_vectors


def train(data, unigram_distribution, word_vectors,
          doc_weights_init=None, n_topics=25,
          batch_size=4096, n_epochs=200,
          lambda_const=100.0, num_sampled=15,
          topics_weight_decay=1e-2,
          topics_lr=1e-3, doc_weights_lr=1e-3, word_vecs_lr=1e-3,
          save_every=10, grad_clip=5.0):
    n_windows = len(data)
    n_documents = len(np.unique(data[:, 0]))
    embedding_dim = word_vectors.shape[1]
    vocab_size = len(unigram_distribution)
    print('number of documents:', n_documents)
    print('number of windows:', n_windows)
    print('number of topics:', n_topics)
    print('vocabulary size:', vocab_size)
    print('word embedding dim:', embedding_dim)

    doc_ids = data[:, 0]
    unique_docs, counts = np.unique(doc_ids, return_counts=True)
    weights = np.zeros((len(unique_docs),), 'float32')
    for i, j in enumerate(unique_docs):
        # longer a document -> lower the document weight when computing loss
        weights[j] = 1.0 / np.log(counts[i])
    weights = torch.FloatTensor(weights).cuda()

    unigram_distribution = torch.FloatTensor(unigram_distribution ** BETA)
    unigram_distribution /= unigram_distribution.sum()
    unigram_distribution = unigram_distribution.cuda()

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
    losses = []
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
                sigma = ETA / epoch ** 0.55
                noise = sigma * Variable(torch.randn(doc_weights_shape).cuda())
                model.doc_weights.weight.grad += noise

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
    method = 'train'

    if method == 'train':
        data = np.load('data.npy')
        unigram_distribution = np.load('unigram_distribution.npy')
        word_vectors = np.load('word_vectors.npy')
        doc_weights_init = np.load('doc_weights_init.npy')

        doc_weights_init = np.log(doc_weights_init + 1e-4)

        temperature = 7.0
        doc_weights_init /= temperature

        num_topics = 12
        batch_size = 2048
        n_epochs = 1000
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
        print('number of topics:', num_topics)
        print('vocabulary size:', vocab_size)
        print('word embedding dim:', embedding_dim)

        doc_ids = data[:, 0]
        unique_docs, counts = np.unique(doc_ids, return_counts=True)
        weights = np.zeros((len(unique_docs),), 'float32')
        for i, j in enumerate(unique_docs):
            weights[j] = 1.0 / np.log(counts[i])
        weights = torch.FloatTensor(weights).cuda()
        unigram_distribution = torch.FloatTensor(unigram_distribution ** BETA)
        unigram_distribution /= unigram_distribution.sum()
        unigram_distribution = unigram_distribution.cuda()

        dataset = SimpleDataset(torch.LongTensor(data))
        iterator = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)

        topics = topic_embedding(num_topics, embedding_dim)
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
        losses = []
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
                sigma = ETA / epoch ** 0.55
                noise = sigma * Variable(torch.randn(doc_weights_shape).cuda())
                model.doc_weights.weight.grad += noise

                for p in model.parameters():
                    p.grad = p.grad.clamp(min=-grad_clip, max=grad_clip)
                optimizer.step()
                n_samples = batch.size(0)
                running_neg_loss += neg_loss.item() * n_samples
                running_dirichlet_loss += dirichlet_loss.item() * n_samples

            losses += [(epoch, running_neg_loss / n_windows, running_dirichlet_loss / n_windows)]
            print('{0:.2f} {1:.2f}'.format(*losses[-1][1:]))
            if epoch % save_every == 0:
                print('\nsaving!\n')
                torch.save(model.state_dict(), str(epoch) + '_epoch_model_state.pytorch')

        _write_training_logs(losses)
        torch.save(model.state_dict(), 'model_state.pytorch')

    if method == 'coherence':
        import numpy as np
        import torch


        def softmax(x):
            e = np.exp(x)
            n = np.sum(e, 1, keepdims=True)
            return e / n


        dataset = pd.read_excel("dataAll.xlsx")
        train_set = []
        docs = dataset['content'].values
        docs = [(i, doc) for i, doc in enumerate(docs)]
        decoder = np.load('decoder.npy', allow_pickle=True)[()]
        doc_decoder = np.load('doc_decoder.npy', allow_pickle=True)[()]
        targets = dataset['target']
        target_names = dataset['target_names']
        targets = np.array([targets[doc_decoder[i]] for i in range(len(doc_decoder))])
        state = torch.load('1000_epoch_model_state.pytorch', map_location=lambda storage, loc: storage)
        num_topics = 12
        doc_weights = state['doc_weights.weight'].cpu().clone().numpy()
        topic_vectors = state['topics.topic_vectors'].cpu().clone().numpy()
        resulted_word_vectors = state['neg.embedding.weight'].cpu().clone().numpy()
        topic_dist = softmax(doc_weights)
        doc_vecs = np.matmul(topic_dist, topic_vectors)
        similarity = np.matmul(topic_vectors, resulted_word_vectors.T)
        most_similar = similarity.argsort(axis=1)[:, -10:]
        saves = []
        averages = []


    if method == 'LDAdata':
        import jieba


        def is_chinese(uchar):
            if u'\u4e00' <= uchar <= u'\u9fa5':
                return True
            else:
                return False


        save = []
        num = 0
        for file in ['水果', '猪瘟', '猪肉', '鸡蛋']:
            num += 1
            data = pd.read_excel("{}.xlsx".format(file))
            print(data.shape)
            print(data.columns)
            for x, y in data.iterrows():
                cont = "{} {}".format(y['weibos'], y['zhuanfa'])
                cont_cut = " ".join([i for i in jieba.lcut(cont) if len(i) > 1 and is_chinese(i)])
                if len(cont_cut.split()) > 2:
                    rows = {"content": cont_cut, 'target_names': file, 'target': num}
                    save.append(rows)

        df = pd.DataFrame(save)
        df.to_excel("dataAll.xlsx", index=None)
        print(df.shape)

    if method == 'LDA':
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel
        from gensim import models
        from sklearn.utils import shuffle
        data = shuffle(pd.read_excel("dataAll.xlsx"))

        print(data.shape)

        train_set = []
        lines = data['content'].values

        for line in lines:
            train_set.append([i for i in line.split()])

        dictionary = Dictionary(train_set)
        corpus = [dictionary.doc2bow(text) for text in train_set]  # 构建稀疏向量
        tfidf = models.TfidfModel(corpus)  # 统计tfidf
        corpus_tfidf = tfidf[corpus]  # 得到每个文本的tfidf向量，稀疏矩阵
        num_topics = 12
        lda_model = LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics, iterations=10)
        top_topics = lda_model.top_topics(corpus, coherence='u_mass', topn=12)
        print(top_topics)

        saves = []
        averages = []
        print_topics = []
        fw = open('lda model topicRestlts.txt', 'w', encoding='utf-8')
