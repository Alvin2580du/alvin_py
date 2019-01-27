import os
import argparse
import logging
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import threading
from torch import nn
import torch.nn.functional as F

start_tok = "<s>"
end_tok = "</s>"
unk_tok = "<unk>"
pad_tok = "<pad>"

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')
parser.add_argument('--n_epochs', type=int, default=5, help='Number of epochs [default: 3]')
parser.add_argument('--n_train', type=int, default=3803900, help='Number of training data (up to 3803957 in gigaword)')
parser.add_argument('--n_valid', type=int, default=189651, help='Number of validation data (up to 189651 in gigaword)')
parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=100, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_0.pkl')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='train.log', filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

model_dir = './models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

torch.manual_seed(1)


def load_data_CNN(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]

            question = ' '.join(q_words)
            document = ' '.join(d_words)

        questions.append(question)
        answers.append(answer)
        documents.append(document)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return (documents, questions, answers)


def my_pad_sequence(batch, pad_value):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_value] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)


class BatchManager:
    def __init__(self, datas, batch_size):
        self.steps = int(len(datas) / batch_size)
        # comment following two lines to neglect the last batch
        if self.steps * batch_size < len(datas):
            self.steps += 1
        self.datas = datas
        self.batch_size = batch_size
        self.bid = 0
        self.buffer = []
        self.s1 = threading.Semaphore(1)
        self.t1 = threading.Thread(target=self.loader, args=())
        self.t1.start()

    def loader(self):
        while True:
            self.s1.acquire()
            batch = list(self.datas[self.bid * self.batch_size: (self.bid + 1) * self.batch_size])
            batch = my_pad_sequence(batch, 3)
            self.bid += 1
            if self.bid == self.steps:
                self.bid = 0
            self.buffer.append(batch)

    def next_batch(self):
        batch = self.buffer.pop()
        self.s1.release()
        return batch


class myCollate:
    def __init__(self, pad_value=3):
        self.pad_value = pad_value

    def collate_fn(self, batch_data):
        batch_data.sort(key=lambda x: len(x), reverse=True)
        batch_data = [torch.tensor(x) for x in batch_data]
        padded = pad_sequence(batch_data, batch_first=True, padding_value=self.pad_value)
        # packed = pack_padded_sequence(padded, lens, batch_first=True)
        return padded

    def __call__(self, batch_data):
        return self.collate_fn(batch_data)


def build_vocab(filelist, vocab_file='vocab.json', min_count=5):
    print("Building vocab with min_count=%d..." % min_count)
    freq = defaultdict(int)
    for file in filelist:
        fin = open(file, "r", encoding="utf8")
        for _, line in enumerate(fin):
            for word in line.strip().split():
                freq[word] += 1
        fin.close()
    print('Number of all words: %d' % len(freq))

    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}
    if unk_tok in freq:
        freq.pop(unk_tok)
    for word in freq:
        if freq[word] > min_count:
            vocab[word] = len(vocab)
    print('Number of filtered words: %d, %f%% ' % (len(vocab), len(vocab) / len(freq) * 100))

    json.dump(vocab, open(vocab_file, 'w'))
    return freq


def load_embedding_vocab(embedding_path):
    fin = open(embedding_path)
    vocab = set([])
    for _, line in enumerate(fin):
        vocab.add(line.split()[0])
    return vocab


def build_vocab_from_embeddings(embedding_path, data_file_list):
    embedding_vocab = load_embedding_vocab(embedding_path)
    vocab = {start_tok: 0, end_tok: 1, unk_tok: 2, pad_tok: 3}

    for file in data_file_list:
        fin = open(file)
        for _, line in enumerate(fin):
            for word in line.split():
                if (word in embedding_vocab) and (word not in vocab):
                    vocab[word] = len(vocab)
    return vocab


def load_data(filename, vocab, n_data=None):
    fin = open(filename, "r", encoding="utf8")
    datas = []
    for idx, line in enumerate(fin):
        if idx == n_data or line == '':
            break
        words = line.strip().split()
        words = ['<s>'] + words + ['</s>']
        sample = [vocab[w if w in vocab else unk_tok] for w in words]
        datas.append(sample)
    return datas


class MyDatasets(Dataset):
    def __init__(self, filename, vocab, n_data=None):
        self.datas = load_data(filename, vocab, n_data)
        self._size = len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return self._size


def getDataLoader(filepath, vocab, n_data, batch_size, num_workers=0):
    dataset = MyDatasets(filepath, vocab, n_data)
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, collate_fn=myCollate(vocab[pad_tok]))
    return loader


class DotAttention(nn.Module):
    """
    Dot attention calculation
    """

    def __init__(self):
        super(DotAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_outs, s_prev):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_outs: the encoder states, in shape [batch, seq_len, dim]
        :param s_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :return: c_t: context vector
        """
        alpha_t = torch.bmm(s_prev.transpose(0, 1), enc_outs.transpose(1, 2))  # [batch, 1, seq_len]
        alpha_t = self.softmax(alpha_t)
        c_t = torch.bmm(alpha_t, enc_outs)  # [batch, 1, dim]
        return c_t


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super(BahdanauAttention, self).__init__()
        self.W_U = nn.Linear((enc_dim + dec_dim), dec_dim)
        self.V = nn.Linear(dec_dim, 1)

    def forward(self, enc_outs, s_prev):
        """
        calculate the context vector c_t, both the input and output are batch first
        :param enc_outs: the encoder states, in shape [batch, seq_len, dim]
        :param s_prev: the previous states of decoder, h_{t-1}, in shape [1, batch, dim]
        :return: c_t: context vector
        """
        s_expanded = s_prev.transpose(0, 1).expand(-1, enc_outs.shape[1], -1)
        cat = torch.cat([enc_outs, s_expanded], dim=-1)
        alpha_t = self.V(torch.tanh(self.W_U(cat))).transpose(1, 2)  # [batch, 1, seq_len]
        e_t = F.softmax(alpha_t, dim=-1)
        c_t = torch.bmm(e_t, enc_outs)  # [batch, 1, dim]
        return c_t


class Model(nn.Module):
    def __init__(self, vocab, out_len=10, emb_dim=32, hid_dim=128):
        super(Model, self).__init__()
        self.out_len = out_len
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.vocab = vocab

        self.softmax = nn.Softmax(dim=-1)
        self.embedding_look_up = nn.Embedding(len(self.vocab), self.emb_dim)

        # encoder (with selective gate)
        self.encoder = nn.GRU(self.emb_dim, self.hid_dim // 2, batch_first=True, bidirectional=True)
        self.encoder = nn.LSTM(self.emb_dim, self.hid_dim // 2, batch_first=True, bidirectional=True)

        self.linear1 = nn.Linear(hid_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()

        # self.attention_layer = DotAttention()
        self.attention_layer = BahdanauAttention(self.hid_dim, self.hid_dim)
        self.enc2dec = nn.Linear(self.hid_dim // 2, self.hid_dim)
        self.decoder = nn.GRU(self.emb_dim + self.hid_dim, self.hid_dim, batch_first=True)

        # maxout
        self.W = nn.Linear(emb_dim, 2 * hid_dim)
        self.U = nn.Linear(hid_dim, 2 * hid_dim)
        self.V = nn.Linear(hid_dim, 2 * hid_dim)

        self.dropout = nn.Dropout(p=0.5)

        self.decoder2vocab = nn.Linear(self.hid_dim, len(self.vocab))

        self.loss_layer = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])

    def init_decoder_hidden(self, hidden):
        hidden = torch.tanh(self.enc2dec(hidden[1]).unsqueeze(0))
        return hidden

    def forward(self, inputs, targets):
        outputs, hidden = self.encode(inputs)
        return outputs, hidden

    def encode(self, inputs):
        embeds = self.embedding_look_up(inputs)
        embeds = self.dropout(embeds)
        outputs, hidden = self.encoder(embeds)

        sn = torch.cat([hidden[0], hidden[1]], dim=-1).view(-1, 1, self.hid_dim)
        sGate = self.sigmoid(self.linear1(outputs) + self.linear2(sn))
        outputs = outputs * sGate
        return outputs, hidden

    def maxout(self, w, c_t, hidden):
        r_t = self.W(w) + self.U(c_t) + self.V(hidden.transpose(0, 1))
        m_t = F.max_pool1d(r_t, kernel_size=2, stride=2)
        return m_t

    def decode(self, word, enc_outs, hidden):
        embeds = self.embedding_look_up(word).view(-1, 1, self.emb_dim)
        embeds = self.dropout(embeds)
        c_t = self.attention_layer(enc_outs, hidden)
        outputs, hidden = self.decoder(torch.cat([c_t, embeds], dim=-1), hidden)
        outputs = self.maxout(embeds, c_t, hidden).squeeze()  # comment this line to remove maxout
        logit = self.decoder2vocab(outputs).squeeze()
        return logit, hidden


def run_batch(valid_x, valid_y, model):
    batch_x = valid_x.next_batch().cuda()
    batch_y = valid_y.next_batch().cuda()

    outputs, hidden = model.encode(batch_x)
    hidden = model.init_decoder_hidden(hidden)

    loss = 0
    for i in range(batch_y.shape[1] - 1):
        logit, hidden = model.decode(batch_y[:, i], outputs, hidden)
        loss += model.loss_layer(logit, batch_y[:, i + 1])  # i+1 to exclude start token
    loss /= batch_y.shape[1]  # batch_y.shape[1] == out_seq_len
    return loss


def train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, epochs=1):
    logging.info("Start to train...")
    n_batches = train_x.steps
    for epoch in range(epochs):
        for idx in range(n_batches):
            optimizer.zero_grad()

            loss = run_batch(train_x, train_y, model)
            loss.backward()  # do not use retain_graph=True
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer.step()
            # scheduler.step()

            if (idx + 1) % 50 == 0:
                train_loss = loss.cpu().detach().numpy()
                with torch.no_grad():
                    valid_loss = run_batch(valid_x, valid_y, model)
                logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
                             % (epoch, idx + 1, train_loss, valid_loss))
            del loss

        model.cpu()
        torch.save(model.state_dict(), os.path.join(model_dir, 'params_%d.pkl' % epoch))
        logging.info('Model saved in dir %s' % model_dir)
        model.cuda()
    # model.embedding_look_up.to(torch.device("cpu"))


def main():
    print(args)

    N_EPOCHS = args.n_epochs
    N_TRAIN = args.n_train
    N_VALID = args.n_valid
    BATCH_SIZE = args.batch_size
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim

    TRAIN_X = 'text.txt'
    TRAIN_Y = 'summary.txt'
    VALID_X = TRAIN_X
    VALID_Y = TRAIN_Y

    vocab_file = "vocab.json"
    if not os.path.exists(vocab_file):
        build_vocab([TRAIN_X, TRAIN_Y], vocab_file)
    vocab = json.load(open(vocab_file))

    train_x = BatchManager(load_data(TRAIN_X, vocab, N_TRAIN), BATCH_SIZE)
    train_y = BatchManager(load_data(TRAIN_Y, vocab, N_TRAIN), BATCH_SIZE)

    valid_x = BatchManager(load_data(VALID_X, vocab, N_VALID), BATCH_SIZE)
    valid_y = BatchManager(load_data(VALID_Y, vocab, N_VALID), BATCH_SIZE)

    # model = Seq2SeqAttention(len(vocab), EMB_DIM, HID_DIM, BATCH_SIZE, vocab, max_trg_len=25).cuda()
    model = Model(vocab, out_len=25, emb_dim=EMB_DIM, hid_dim=HID_DIM).cuda()
    # model.embedding_look_up.to(torch.device("cpu"))

    model_file = args.model_file
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
        logging.info('Load model parameters from %s' % model_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.3)
    train(train_x, train_y, valid_x, valid_y, model, optimizer, scheduler, N_EPOCHS)


if __name__ == '__main__':
    main()
