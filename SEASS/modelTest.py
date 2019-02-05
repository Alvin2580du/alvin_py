import os
import json
import torch
import argparse
import numpy as np
import threading
import torch.nn.functional as F
from torch import nn

parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in pytorch')

parser.add_argument('--n_valid', type=int, default=725,
                    help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
parser.add_argument('--input_file', type=str, default="clean_data/test.source.cut", help='input file')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 32]')
parser.add_argument('--emb_dim', type=int, default=100, help='Embedding size [default: 256]')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--model_file', type=str, default='./models/params_idx_2499.pkl', help='model file path')
parser.add_argument('--search', type=str, default='greedy', help='greedy/beam')
parser.add_argument('--beam_width', type=int, default=12, help='beam search width')
args = parser.parse_args()
print(args)

if not os.path.exists(args.model_file):
    raise FileNotFoundError("model file not found")

start_tok = "<s>"
end_tok = "</s>"
unk_tok = "<unk>"
pad_tok = "<pad>"


def my_pad_sequence(batch, pad_value):
    max_len = max([len(b) for b in batch])
    batch = [b + [pad_value] * (max_len - len(b)) for b in batch]
    return torch.tensor(batch)


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
        # self.encoder = nn.LSTM(self.emb_dim, self.hid_dim // 2, batch_first=True, bidirectional=True)

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


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, hidden, device=torch.device("cuda:0")):
        """Initialize params."""
        self.size = size
        self.done = False
        # self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.device = device
        self.tt = torch.cuda if device.type == "cuda" else torch
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size, device=self.device).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size, device=self.device).fill_(self.eos)]
        self.nextYs[0][0] = self.bos

        # the hidden state at current time-step
        hidden = hidden.view(1, 1, -1)
        self.hidden = hidden.expand((1, size, hidden.shape[2]))

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_word(self):
        """Get state of beam."""
        return self.nextYs[-1]

    def get_hidden_state(self):
        return self.hidden.contiguous()

    # Get the backpointers for the current timestep.
    def get_prev_word(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given log_prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance_(self, log_probs, hidden):
        if self.done:
            return True

        """Advance the beam."""
        log_probs = log_probs.squeeze()  # k*V
        num_words = log_probs.shape[-1]

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = log_probs + self.scores.unsqueeze(1).expand_as(log_probs)
        else:
            beam_lk = log_probs[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        self.hidden = hidden[:, prev_k, :]  # hidden: 1 * k * hid_dim

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]


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


def print_summaries(summaries, vocab):
    """
    param summaries: in shape (seq_len, batch)
    """
    i2w = {key: value for value, key in vocab.items()}
    fout = open("SummaryResults.txt", "w")

    for idx in range(len(summaries)):
        line = [i2w[tok] for tok in summaries[idx] if tok != vocab["</s>"]]
        fout.writelines(" ".join(line) + "\n")
    fout.close()


def greedy(model, batch_x, max_trg_len=10):
    enc_outs, hidden = model.encode(batch_x)
    hidden = model.init_decoder_hidden(hidden)
    words = []
    word = torch.ones(hidden.shape[1], dtype=torch.long) * model.vocab["<s>"]
    for _ in range(max_trg_len):
        logit, hidden = model.decode(word, enc_outs, hidden)
        word = torch.argmax(logit, dim=-1)
        words.append(word.cpu().numpy())
    return np.array(words).T


def beam_search(model, batch_x, max_trg_len=10, k=args.beam_width):
    enc_outs, hidden = model.encode(batch_x)
    hidden = model.init_decoder_hidden(hidden)

    beams = [Beam(k, model.vocab, hidden[:, i, :])
             for i in range(batch_x.shape[0])]

    for _ in range(max_trg_len):
        for j in range(len(beams)):
            hidden = beams[j].get_hidden_state()
            word = beams[j].get_current_word()
            enc_outs_j = enc_outs[j].unsqueeze(0).expand(k, -1, -1)
            logit, hidden = model.decode(word, enc_outs_j, hidden)
            # logit: [k x V], hidden: [k x hid_dim]
            log_probs = F.softmax(logit, -1)
            beams[j].advance_(log_probs, hidden)

    allHyp, allScores = [], []
    n_best = 1
    for b in range(batch_x.shape[0]):
        scores, ks = beams[b].sort_best()
        allScores += [scores[:n_best]]
        hyps = [beams[b].get_hyp(k) for k in ks[:n_best]]
        allHyp.append(hyps)

    # shape of allHyp: [batch, 1, list]
    allHyp = [[int(w.cpu().numpy()) for w in hyp[0]] for hyp in allHyp]
    return allHyp


def my_test(valid_x, model):
    summaries = []
    with torch.no_grad():
        for _ in range(valid_x.steps):
            batch_x = valid_x.next_batch()
            if args.search == "greedy":
                summary = greedy(model, batch_x)
            elif args.search == "beam":
                summary = beam_search(model, batch_x)
            else:
                raise NameError("Unknown search method")
            summaries.extend(summary)
    print_summaries(summaries, model.vocab)
    print("Done!")


def main():
    N_VALID = args.n_valid
    BATCH_SIZE = args.batch_size
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim

    vocab = json.load(open('vocab_zh.json'))
    valid_x = BatchManager(load_data(args.input_file, vocab, N_VALID), BATCH_SIZE)

    model = Model(vocab, out_len=15, emb_dim=EMB_DIM, hid_dim=HID_DIM)
    model.eval()

    file = args.model_file
    if os.path.exists(file):
        model.load_state_dict(torch.load(file))
        print('Load model parameters from %s' % file)

    my_test(valid_x, model)


if __name__ == '__main__':
    main()
