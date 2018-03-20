import os
import time
import pickle
import collections
import string
import numpy as np
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.contrib.rnn import GRUCell
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string('LOGS_DIR', './logs/', '')
flags.DEFINE_string('DATA_DIR', './data/1-billion-word-language-modeling-benchmark-r13output/', "")
flags.DEFINE_string('CKPT_PATH', "./ckpt/", "")
flags.DEFINE_integer('BATCH_SIZE', 64, '')
flags.DEFINE_integer('CRITIC_ITERS', 10, '')
flags.DEFINE_integer('LAMBDA', 10, '')
flags.DEFINE_integer('MAX_N_EXAMPLES', 10000000, '')
flags.DEFINE_string('GENERATOR_MODEL', 'Generator_GRU_CL_VL_TH', '')
flags.DEFINE_string('DISCRIMINATOR_MODEL', 'Discriminator_GRU', '')
flags.DEFINE_string('PICKLE_PATH', './pkl', '')
flags.DEFINE_integer('GEN_ITERS', 50, '')
flags.DEFINE_integer('ITERATIONS_PER_SEQ_LENGTH', 15000, '')
flags.DEFINE_float('NOISE_STDEV', 10.0, '')
flags.DEFINE_integer('DISC_STATE_SIZE', 512, '')
flags.DEFINE_integer('GEN_STATE_SIZE', 512, '')
flags.DEFINE_boolean('TRAIN_FROM_CKPT', False, '')
flags.DEFINE_integer('GEN_GRU_LAYERS', 1, '')
flags.DEFINE_integer('DISC_GRU_LAYERS', 1, '')
flags.DEFINE_integer('START_SEQ', 1, '')
flags.DEFINE_integer('END_SEQ', 32, '')
flags.DEFINE_bool('PADDING_IS_SUFFIX', False, '')
flags.DEFINE_integer('SAVE_CHECKPOINTS_EVERY', 25000, '')
flags.DEFINE_boolean('LIMIT_BATCH', True, '')
flags.DEFINE_boolean('SCHEDULE_ITERATIONS', False, '')
flags.DEFINE_integer('SCHEDULE_MULT', 2000, '')
flags.DEFINE_boolean('DYNAMIC_BATCH', False, '')
flags.DEFINE_string('SCHEDULE_SPEC', 'all', '')

# Only for inference mode

flags.DEFINE_string('INPUT_SAMPLE', './output/sample.txt', '')

FLAGS = flags.FLAGS

# only for backward compatability

LOGS_DIR = os.path.join(FLAGS.LOGS_DIR, "%s-%s-%s-%s-%s-%s-%s-" % (FLAGS.GENERATOR_MODEL, FLAGS.DISCRIMINATOR_MODEL,
                                                   FLAGS.GEN_ITERS, FLAGS.CRITIC_ITERS,
                                                   FLAGS.DISC_STATE_SIZE, FLAGS.GEN_STATE_SIZE,
                                                   time.time()))


class RestoreConfig():
    def __init__(self):
        if FLAGS.TRAIN_FROM_CKPT:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=False)
        else:
            self.restore_dir = self.set_restore_dir(load_from_curr_session=True)

    def set_restore_dir(self, load_from_curr_session=True):
        if load_from_curr_session:
            restore_dir = os.path.join(LOGS_DIR, 'checkpoint')
        else:
            restore_dir = FLAGS.CKPT_PATH
        return restore_dir

    def get_restore_dir(self):
        return self.restore_dir


def create_logs_dir():
    os.makedirs(LOGS_DIR)


restore_config = RestoreConfig()
DATA_DIR = FLAGS.DATA_DIR
BATCH_SIZE = FLAGS.BATCH_SIZE
CRITIC_ITERS = FLAGS.CRITIC_ITERS
LAMBDA = FLAGS.LAMBDA
MAX_N_EXAMPLES = FLAGS.MAX_N_EXAMPLES
PICKLE_PATH = FLAGS.PICKLE_PATH
PICKLE_LOAD = True
CKPT_PATH = FLAGS.CKPT_PATH
GENERATOR_MODEL = FLAGS.GENERATOR_MODEL
DISCRIMINATOR_MODEL = FLAGS.DISCRIMINATOR_MODEL
GEN_ITERS = FLAGS.GEN_ITERS

CHARMAP_FN = 'charmap_32.pkl'
INV_CHARMAP_FN = 'inv_charmap_32.pkl'
INV_CHARMAP_PKL_PATH = PICKLE_PATH + '/' + INV_CHARMAP_FN
CHARMAP_PKL_PATH = PICKLE_PATH + '/' + CHARMAP_FN


def tokenize_string(sample):
    return tuple(sample.lower().split(' '))


class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample) - n + 1):
                yield sample[i:i + n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        else:
            return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        # p is another NgramLanguageModel
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        # p is another NgramLanguageModel
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i ** 2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i ** 2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        # p is another NgramLanguageModel
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5 * (kl_p_m + kl_q_m) / np.log(2)


def replace_trash(unicode_string):
    printable = set(string.printable)
    return filter(lambda x: x in printable, unicode_string)


def load_dataset(max_length, max_n_examples, tokenize=False, max_vocab_size=2048,
                 data_dir='/home/ishaan/data/1-billion-word-language-modeling-benchmark-r13output',
                 pad=True, dataset='training'):
    assert dataset == 'training' or dataset == 'heldout', "only available datasets are 'training' and 'heldout'"
    lines = []

    finished = False
    number_of_divided_files = 100 if dataset == 'training' else 50

    for i in range(number_of_divided_files - 1):
        path = data_dir + ("/{}-monolingual.tokenized.shuffled/news.en{}-{}-of-{}".format(dataset,
                                                                                          '' if dataset == 'training' else '.heldout',
                                                                                          str(i + 1).zfill(5),
                                                                                          str(
                                                                                              number_of_divided_files).zfill(
                                                                                              5)))
        with open(path, 'r') as f:
            for line in f:
                line = line[:max_length]

                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > max_length:
                    line = line[:max_length]

                if pad:
                    line = line + (("`",) * (max_length - len(line)))

                lines.append(line)

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    import collections
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {'unk': 0}
    inv_charmap = ['unk']

    for char, count in counts.most_common(10000000):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    # for i in range(100):
    #     print(filtered_lines[i])

    print("loaded {} lines in dataset".format(len(lines)))
    return filtered_lines, charmap, inv_charmap


def generate_argmax_samples_and_gt_samples(session, inv_charmap, fake_inputs, disc_fake, gen, real_inputs_discrete,
                                           feed_gt=True):
    scores = []
    samples = []
    samples_probabilites = []
    for i in range(10):
        argmax_samples, real_samples, samples_scores = generate_samples(session, inv_charmap, fake_inputs, disc_fake,
                                                                        gen, real_inputs_discrete, feed_gt=feed_gt)
        samples.extend(argmax_samples)
        scores.extend(samples_scores)
        samples_probabilites.extend(real_samples)
    return samples, samples_probabilites, scores


def generate_samples(session, inv_charmap, fake_inputs, disc_fake, gen, real_inputs_discrete, feed_gt=True):
    # sampled data here is only to calculate loss
    if feed_gt:
        f_dict = {real_inputs_discrete: next(gen)}
    else:
        f_dict = {}

    fake_samples, fake_scores = session.run([fake_inputs, disc_fake], feed_dict=f_dict)
    fake_scores = np.squeeze(fake_scores)

    decoded_samples = decode_indices_to_string(np.argmax(fake_samples, axis=2), inv_charmap)
    return decoded_samples, fake_samples, fake_scores


def decode_indices_to_string(samples, inv_charmap):
    decoded_samples = []
    for i in range(len(samples)):
        decoded = []
        for j in range(len(samples[i])):
            decoded.append(inv_charmap[samples[i][j]])
        decoded_samples.append(tuple(decoded))
    return decoded_samples


def inf_train_gen(lines, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + BATCH_SIZE]],
                dtype='int32'
            )


def load_picklized(path):
    with open(path, 'rb') as f:
        content = pickle.load(f)
        f.close()
    return content


def save_picklized(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dataset_from_pkl(b_lines, b_charmap, b_inv_charmap, lines_pkl_path):
    if b_lines:
        lines = load_picklized(lines_pkl_path)
    else:
        lines = None

    if b_charmap:
        charmap = load_picklized(CHARMAP_PKL_PATH)
    else:
        charmap = None

    if b_inv_charmap:
        inv_charmap = load_picklized(INV_CHARMAP_PKL_PATH)
    else:
        inv_charmap = None

    return lines, charmap, inv_charmap


def load_dataset_v1(b_lines=True, b_charmap=True, b_inv_charmap=True,
                    seq_length=32, n_examples=10000000, tokenize=False,
                    pad=True, dataset='training'):
    LINES_FN = 'lines_%s_%s.pkl' % (seq_length, tokenize)
    if dataset != 'training':
        LINES_FN = dataset + '_' + LINES_FN
    LINES_PKL_PATH = PICKLE_PATH + '/' + LINES_FN

    if PICKLE_PATH is not None and PICKLE_LOAD is True and (
            b_lines is False or (b_lines and os.path.exists(LINES_PKL_PATH))) \
            and (b_charmap is False or (b_charmap and os.path.exists(CHARMAP_PKL_PATH))) and \
            (b_inv_charmap is False or (b_inv_charmap and os.path.exists(INV_CHARMAP_PKL_PATH))):

        print("Loading lines, charmap, inv_charmap from pickle files")
        lines, charmap, inv_charmap = load_dataset_from_pkl(b_lines=b_lines, b_charmap=b_charmap,
                                                            b_inv_charmap=b_inv_charmap, lines_pkl_path=LINES_PKL_PATH)

    else:
        print("Loading lines, charmap, inv_charmap from Dataset & Saving to pickle")
        lines, charmap, inv_charmap = load_dataset(
            max_length=seq_length,
            max_n_examples=n_examples,
            data_dir=DATA_DIR,
            tokenize=tokenize,
            pad=pad,
            dataset=dataset
        )

        # save to pkl
        if not os.path.isdir(PICKLE_PATH):
            os.mkdir(PICKLE_PATH)

        if b_lines:
            save_picklized(lines, LINES_PKL_PATH)
        if b_charmap:
            save_picklized(charmap, CHARMAP_PKL_PATH)
        if b_inv_charmap:
            save_picklized(inv_charmap, INV_CHARMAP_PKL_PATH)

    return lines, charmap, inv_charmap


def get_internal_checkpoint_dir(seq_length):
    internal_checkpoint_dir = os.path.join(restore_config.get_restore_dir(), "seq-%d" % seq_length)
    if not os.path.isdir(internal_checkpoint_dir):
        os.makedirs(internal_checkpoint_dir)
    return internal_checkpoint_dir


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print("Not loading: %s." % saved_var_name)
    saver = tf.train.Saver(restore_vars)
    print("Restoring vars:")
    print(restore_vars)
    saver.restore(session, save_file)


def Discriminator_GRU(inputs, charmap_len, seq_len, reuse=False):
    with tf.variable_scope("Discriminator", reuse=reuse):
        num_neurons = FLAGS.DISC_STATE_SIZE

        weight = tf.get_variable("embedding", shape=[charmap_len, num_neurons],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        # backwards compatability
        if FLAGS.DISC_GRU_LAYERS == 1:
            cell = GRUCell(num_neurons)
        else:
            cell = tf.contrib.rnn.MultiRNNCell([GRUCell(num_neurons) for _ in range(FLAGS.DISC_GRU_LAYERS)],
                                               state_is_tuple=True)

        flat_inputs = tf.reshape(inputs, [-1, charmap_len])

        inputs = tf.reshape(tf.matmul(flat_inputs, weight), [-1, seq_len, num_neurons])
        inputs = tf.unstack(tf.transpose(inputs, [1, 0, 2]))

        for inp in inputs:
            print(inp.get_shape())

        output, state = tf.contrib.rnn.static_rnn(
            cell,
            inputs,
            dtype=tf.float32
        )

        last = output[-1]

        weight = tf.get_variable("W", shape=[num_neurons, 1],
                                 initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        bias = tf.get_variable("b", shape=[1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

        prediction = tf.matmul(last, weight) + bias

        return prediction


def Generator_GRU_CL_VL_TH(n_samples, charmap_len, seq_len=None, gt=None):
    with tf.variable_scope("Generator"):
        noise, noise_shape = get_noise()
        num_neurons = FLAGS.GEN_STATE_SIZE

        cells = []
        for l in range(FLAGS.GEN_GRU_LAYERS):
            cells.append(GRUCell(num_neurons))

        # this is separate to decouple train and test
        train_initial_states = create_initial_states(noise)
        inference_initial_states = create_initial_states(noise)

        sm_weight = tf.Variable(tf.random_uniform([num_neurons, charmap_len], minval=-0.1, maxval=0.1))
        sm_bias = tf.Variable(tf.random_uniform([charmap_len], minval=-0.1, maxval=0.1))

        embedding = tf.Variable(tf.random_uniform([charmap_len, num_neurons], minval=-0.1, maxval=0.1))

        char_input = tf.Variable(tf.random_uniform([num_neurons], minval=-0.1, maxval=0.1))
        char_input = tf.reshape(tf.tile(char_input, [n_samples]), [n_samples, 1, num_neurons])

        if seq_len is None:
            seq_len = tf.placeholder(tf.int32, None, name="ground_truth_sequence_length")

        if gt is not None:  # if no GT, we are training
            train_pred = get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len,
                                      sm_bias, sm_weight, train_initial_states)
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight,
                                            inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=True)
        else:
            inference_op = get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight,
                                            inference_initial_states,
                                            num_neurons,
                                            charmap_len, reuse=False)
            train_pred = None

        return train_pred, inference_op


def create_initial_states(noise):
    states = []
    for l in range(FLAGS.GEN_GRU_LAYERS):
        states.append(noise)
    return states


def get_train_op(cells, char_input, charmap_len, embedding, gt, n_samples, num_neurons, seq_len, sm_bias, sm_weight,
                 states):
    gt_embedding = tf.reshape(gt, [n_samples * seq_len, charmap_len])
    gt_GRU_input = tf.matmul(gt_embedding, embedding)
    gt_GRU_input = tf.reshape(gt_GRU_input, [n_samples, seq_len, num_neurons])[:, :-1]
    gt_sentence_input = tf.concat([char_input, gt_GRU_input], axis=1)
    GRU_output, _ = rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias,
                                        sm_weight,
                                        states)
    train_pred = []
    # TODO: optimize loop
    for i in range(seq_len):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_len - i - 1, charmap_len]), gt[:, :i], GRU_output[:, i:i + 1, :]],
                      axis=1))

    train_pred = tf.reshape(train_pred, [BATCH_SIZE * seq_len, seq_len, charmap_len])

    if FLAGS.LIMIT_BATCH:
        indices = tf.random_uniform([BATCH_SIZE], 0, BATCH_SIZE * seq_len, dtype=tf.int32)
        train_pred = tf.gather(train_pred, indices)

    return train_pred


def rnn_step_prediction(cells, charmap_len, gt_sentence_input, num_neurons, seq_len, sm_bias, sm_weight, states,
                        reuse=False):
    with tf.variable_scope("rnn", reuse=reuse):
        GRU_output = gt_sentence_input
        for l in range(FLAGS.GEN_GRU_LAYERS):
            GRU_output, states[l] = tf.nn.dynamic_rnn(cells[l], GRU_output, dtype=tf.float32,
                                                      initial_state=states[l], scope="layer_%d" % (l + 1))
    GRU_output = tf.reshape(GRU_output, [-1, num_neurons])
    GRU_output = tf.nn.softmax(tf.matmul(GRU_output, sm_weight) + sm_bias)
    GRU_output = tf.reshape(GRU_output, [BATCH_SIZE, -1, charmap_len])
    return GRU_output, states


def get_inference_op(cells, char_input, embedding, seq_len, sm_bias, sm_weight, states, num_neurons, charmap_len,
                     reuse=False):
    inference_pred = []
    embedded_pred = [char_input]
    for i in range(seq_len):
        step_pred, states = rnn_step_prediction(cells, charmap_len, tf.concat(embedded_pred, 1), num_neurons, seq_len,
                                                sm_bias, sm_weight, states, reuse=reuse)
        best_chars_tensor = tf.argmax(step_pred, axis=2)
        best_chars_one_hot_tensor = tf.one_hot(best_chars_tensor, charmap_len)
        best_char = best_chars_one_hot_tensor[:, -1, :]
        inference_pred.append(tf.expand_dims(best_char, 1))
        embedded_pred.append(tf.expand_dims(tf.matmul(best_char, embedding), 1))
        reuse = True  # no matter what the reuse was, after the first step we have to reuse the defined vars

    return tf.concat(inference_pred, axis=1)


generators = {
    "Generator_GRU_CL_VL_TH": Generator_GRU_CL_VL_TH,
}

discriminators = {
    "Discriminator_GRU": Discriminator_GRU,
}


def get_noise():
    noise_shape = [BATCH_SIZE, FLAGS.GEN_STATE_SIZE]
    return make_noise(shape=noise_shape, stddev=FLAGS.NOISE_STDEV), noise_shape


def get_generator(model_name):
    return generators[model_name]


def params_with_name(name):
    return [p for p in tf.trainable_variables() if name in p.name]


def get_discriminator(model_name):
    return discriminators[model_name]


def make_noise(shape, mean=0.0, stddev=1.0):
    return tf.random_normal(shape, mean, stddev)


def get_optimization_ops(disc_cost, gen_cost, global_step):
    gen_params = params_with_name('Generator')
    disc_params = params_with_name('Discriminator')
    print("Generator Params: %s" % gen_params)
    print("Disc Params: %s" % disc_params)
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                             var_list=gen_params,
                                                                                             global_step=global_step)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                              var_list=disc_params)
    return disc_train_op, gen_train_op


def get_substrings_from_gt(real_inputs, seq_length, charmap_len):
    train_pred = []
    for i in range(seq_length):
        train_pred.append(
            tf.concat([tf.zeros([BATCH_SIZE, seq_length - i - 1, charmap_len]), real_inputs[:, :i + 1]],
                      axis=1))

    all_sub_strings = tf.reshape(train_pred, [BATCH_SIZE * seq_length, seq_length, charmap_len])

    if FLAGS.LIMIT_BATCH:
        indices = tf.random_uniform([BATCH_SIZE], 1, all_sub_strings.get_shape()[0], dtype=tf.int32)
        all_sub_strings = tf.gather(all_sub_strings, indices)
        return all_sub_strings[:BATCH_SIZE]
    else:
        return all_sub_strings


def define_objective(charmap, real_inputs_discrete, seq_length):
    real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
    Generator = get_generator(FLAGS.GENERATOR_MODEL)
    Discriminator = get_discriminator(FLAGS.DISCRIMINATOR_MODEL)
    train_pred, inference_op = Generator(BATCH_SIZE, len(charmap), seq_len=seq_length, gt=real_inputs)

    real_inputs_substrings = get_substrings_from_gt(real_inputs, seq_length, len(charmap))

    disc_real = Discriminator(real_inputs_substrings, len(charmap), seq_length, reuse=False)
    disc_fake = Discriminator(train_pred, len(charmap), seq_length, reuse=True)
    disc_on_inference = Discriminator(inference_op, len(charmap), seq_length, reuse=True)

    disc_cost, gen_cost = loss_d_g(disc_fake, disc_real, train_pred, real_inputs_substrings, charmap, seq_length,
                                   Discriminator)
    return disc_cost, gen_cost, train_pred, disc_fake, disc_real, disc_on_inference, inference_op


def loss_d_g(disc_fake, disc_real, fake_inputs, real_inputs, charmap, seq_length, Discriminator):
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[tf.shape(real_inputs)[0], 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha * differences)
    gradients = tf.gradients(Discriminator(interpolates, len(charmap), seq_length, reuse=True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += LAMBDA * gradient_penalty

    return disc_cost, gen_cost


def define_summaries(disc_cost, gen_cost, seq_length):
    train_writer = tf.summary.FileWriter(LOGS_DIR)
    tf.summary.scalar("d_loss_%d" % seq_length, disc_cost)
    tf.summary.scalar("g_loss_%d" % seq_length, gen_cost)
    merged = tf.summary.merge_all()
    return merged, train_writer


def log_samples(samples, scores, iteration, seq_length, prefix):
    sample_scores = zip(samples, scores)
    sample_scores = sorted(sample_scores, key=lambda sample: sample[1])

    with open(get_internal_checkpoint_dir(seq_length) + '/{}_samples_{}.txt'.format(
            prefix, iteration),
              'a') as f:
        for s, score in sample_scores:
            s = "".join(s)
            f.write("%s||\t%f\n" % (s, score))
    f.close()


def log_run_settings():
    with open(os.path.join(LOGS_DIR, 'run_settings.txt'), 'w') as f:
        for key in tf.flags.FLAGS.__flags.keys():
            entry = "%s: %s" % (key, tf.flags.FLAGS.__flags[key])
            f.write(entry + '\n')
            print(entry)
    f.close()


def get_grams_cached(lines):
    grams_filename = FLAGS.PICKLE_PATH + '/true-char-ngrams.pkl'
    if os.path.exists(grams_filename):
        return load_picklized(grams_filename)
    else:
        grams = get_grams(lines)
        save_picklized(grams, grams_filename)
        return grams


def get_grams(lines):
    lines_joined = [''.join(l) for l in lines]

    unigrams = dict()
    bigrams = dict()
    trigrams = dict()
    quadgrams = dict()
    token_count = 0

    for l in lines_joined:
        l = l.split(" ")
        l = filter(lambda x: x != ' ' and x != '', l)

        for i in range(len(l)):
            token_count += 1
            unigrams[l[i]] = unigrams.get(l[i], 0) + 1
            if i >= 1:
                bigrams[(l[i - 1], l[i])] = bigrams.get((l[i - 1], l[i]), 0) + 1
            if i >= 2:
                trigrams[(l[i - 2], l[i - 1], l[i])] = trigrams.get((l[i - 2], l[i - 1], l[i]), 0) + 1
            if i >= 3:
                quadgrams[(l[i - 3], l[i - 2], l[i - 1], l[i])] = quadgrams.get((l[i - 3], l[i - 2], l[i - 1], l[i]),
                                                                                0) + 1

    return unigrams, bigrams, trigrams, quadgrams


def percentage_real(samples_grams, real_grams):
    grams_in_real = 0

    for g in samples_grams:
        if g in real_grams:
            grams_in_real += 1
    if len(samples_grams) > 0:
        return grams_in_real * 1.0 / len(samples_grams)
    return 0


def percentage_startswith(e_samples, unigrams_real):
    counter = 0
    for prefix in e_samples:
        for uni in unigrams_real:
            if uni.startswith(prefix):
                counter += 1
                break
    # print counter
    return counter * 1.0 / len(e_samples.keys())


def run(iterations, seq_length, is_first, charmap, inv_charmap, prev_seq_length):
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in single_length_train.py!')

    lines, _, _ = load_dataset_v1(seq_length=seq_length, b_charmap=False, b_inv_charmap=False,
                                  n_examples=FLAGS.MAX_N_EXAMPLES)

    real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, seq_length])

    global_step = tf.Variable(0, trainable=False)
    disc_cost, gen_cost, fake_inputs, disc_fake, disc_real, disc_on_inference, inference_op = define_objective(charmap,
                                                                                                               real_inputs_discrete,
                                                                                                               seq_length)
    merged, train_writer = define_summaries(disc_cost, gen_cost, seq_length)
    disc_train_op, gen_train_op = get_optimization_ops(disc_cost, gen_cost, global_step)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as session:

        session.run(tf.initialize_all_variables())
        if not is_first:
            print("Loading previous checkpoint...")
            internal_checkpoint_dir = get_internal_checkpoint_dir(prev_seq_length)
            optimistic_restore(session,
                               latest_checkpoint(internal_checkpoint_dir, "checkpoint"))

            restore_config.set_restore_dir(load_from_curr_session=True)
            # global param, always load from curr session after finishing the first seq

        gen = inf_train_gen(lines, charmap)

        for iteration in range(iterations):
            start_time = time.time()

            # Train critic
            for i in range(CRITIC_ITERS):
                _data = next(gen)
                _disc_cost, _, real_scores = session.run(
                    [disc_cost, disc_train_op, disc_real],
                    feed_dict={real_inputs_discrete: _data}
                )

            # Train G
            for i in range(GEN_ITERS):
                _data = next(gen)
                _ = session.run(gen_train_op, feed_dict={real_inputs_discrete: _data})

            print("iteration %s/%s" % (iteration, iterations))
            print("disc cost %f" % _disc_cost)

            # Summaries
            if iteration % 100 == 99:
                _data = next(gen)
                summary_str = session.run(
                    merged,
                    feed_dict={real_inputs_discrete: _data}
                )

                train_writer.add_summary(summary_str, global_step=iteration)
                fake_samples, samples_real_probabilites, fake_scores = generate_argmax_samples_and_gt_samples(session,
                                                                                                              inv_charmap,
                                                                                                              fake_inputs,
                                                                                                              disc_fake,
                                                                                                              gen,
                                                                                                              real_inputs_discrete,
                                                                                                              feed_gt=True)

                log_samples(fake_samples, fake_scores, iteration, seq_length, "train")
                log_samples(decode_indices_to_string(_data, inv_charmap), real_scores, iteration, seq_length,
                            "gt")
                test_samples, _, fake_scores = generate_argmax_samples_and_gt_samples(session, inv_charmap,
                                                                                      inference_op,
                                                                                      disc_on_inference,
                                                                                      gen,
                                                                                      real_inputs_discrete,
                                                                                      feed_gt=False)
                # disc_on_inference, inference_op
                log_samples(test_samples, fake_scores, iteration, seq_length, "test")

            if iteration % FLAGS.SAVE_CHECKPOINTS_EVERY == FLAGS.SAVE_CHECKPOINTS_EVERY - 1:
                saver.save(session, get_internal_checkpoint_dir(seq_length) + "/ckp")

        saver.save(session, get_internal_checkpoint_dir(seq_length) + "/ckp")
        session.close()


# ------------------------------ train ---------------------------- #

def build_run():
    REAL_BATCH_SIZE = FLAGS.BATCH_SIZE

    if FLAGS.SCHEDULE_SPEC == 'all':
        stages = range(FLAGS.START_SEQ, FLAGS.END_SEQ)
    else:
        split = FLAGS.SCHEDULE_SPEC.split(',')
        stages = map(int, split)

    print('@@@@@@@@@@@ Stages : ' + ' '.join(map(str, stages)))

    _, charmap, inv_charmap = load_dataset_v1(seq_length=32, b_lines=False)

    for i in range(len(stages)):
        prev_seq_length = stages[i - 1] if i > 0 else 0
        seq_length = stages[i]
        print("********************Training on Seq Len = %d, BATCH SIZE: %d**********************" % (
            seq_length, BATCH_SIZE))
        tf.reset_default_graph()
        if FLAGS.SCHEDULE_ITERATIONS:
            iterations = min((seq_length + 1) * FLAGS.SCHEDULE_MULT, FLAGS.ITERATIONS_PER_SEQ_LENGTH)
        else:
            iterations = FLAGS.ITERATIONS_PER_SEQ_LENGTH
        run(iterations, seq_length, seq_length == stages[0] and not (FLAGS.TRAIN_FROM_CKPT),
            charmap,
            inv_charmap,
            prev_seq_length)

        if FLAGS.DYNAMIC_BATCH:
            BATCH_SIZE = REAL_BATCH_SIZE / seq_length


# ------------------------------ generate ---------------------------- #

def generate():
    output_path = './output/sample.txt'

    '''
    example usage:
    python generate.py --CKPT_PATH=/path/to/checkpoint/seq-32/ckp --DISC_GRU_LAYERS=2 --GEN_GRU_LAYERS=2
    '''

    SEQ_LEN = FLAGS.END_SEQ

    _, charmap, inv_charmap = load_dataset_v1()
    charmap_len = len(charmap)

    Generator = get_generator(GENERATOR_MODEL)
    Discriminator = get_discriminator(DISCRIMINATOR_MODEL)

    _, inference_op = Generator(BATCH_SIZE, charmap_len, seq_len=SEQ_LEN)
    disc_fake = Discriminator(inference_op, charmap_len, SEQ_LEN, reuse=False)

    saver = tf.train.Saver()

    with tf.Session() as session:
        saver.restore(session, CKPT_PATH)
        sequential_output, scores = session.run([inference_op, disc_fake])

    samples = []

    for i in range(BATCH_SIZE):
        chars = []
        for seq_len in range(sequential_output.shape[1]):
            char_index = np.argmax(sequential_output[i, seq_len])
            chars.append(inv_charmap[char_index])
        sample = "".join(chars)
        samples.append(sample)

    if not (os.path.isdir('./output')):
        os.mkdir("./output")

    with open(output_path, 'w') as f:
        for k in samples:
            f.write("%s\n" % k)
    f.close()
    print("Prediction saved to: %s" % output_path)


# ------------------------------ evaluate ---------------------------- #

def evaluate(input_sample, gt_grams):
    # char level evaluation

    sample_lines = load_sample(input_sample, tokenize=False)

    u_samples, b_samples, t_samples, q_samples = get_grams(sample_lines)
    u_real, b_real, t_real, q_real = gt_grams

    print("Unigrams: %f" % percentage_real(u_samples, u_real))
    print("Bigrams: %f" % percentage_real(b_samples, b_real))
    print("Trigrams: %f" % percentage_real(t_samples, t_real))
    print("Quad grams: %f" % percentage_real(q_samples, q_real))


def cut_end_words(lines):
    lines_without_ends = []
    for l in lines:
        lines_without_ends.append(l[:-1])

    return lines_without_ends


def load_sample(input_sample, tokenize=False):
    with open(input_sample, 'r') as f:
        lines_samples = [l for l in f]
    f.close()
    preprocessed_lines = preprocess_sample(lines_samples, '\n', tokenize=tokenize)
    return preprocessed_lines


def load_gt(tokenize=False, dataset='heldout'):
    # test on char level
    print("Loading lines, charmap, inv_charmap")
    lines, _, _ = load_dataset_v1(
        b_lines=True,
        b_charmap=False,
        b_inv_charmap=False,
        n_examples=FLAGS.MAX_N_EXAMPLES,
        tokenize=tokenize,
        pad=False,
        dataset=dataset
    )

    return lines


def preprocess_sample(lines_samples, separator, tokenize):
    preprocessed_lines = []
    for line in lines_samples:
        if separator is not None:
            line = separator.join(line.split(separator)[:-1])

        if tokenize:
            line = tokenize_string(line)
        else:
            line = tuple(line)

        preprocessed_lines.append(line)
    return preprocessed_lines


def get_gt_grams_cached(lines, dataset='training'):
    grams_filename = 'true-char-ngrams.pkl'
    if dataset == 'heldout':
        grams_filename = 'heldout_' + grams_filename
    grams_filename = FLAGS.PICKLE_PATH + '/' + grams_filename
    if os.path.exists(grams_filename):
        return load_picklized(grams_filename)
    else:
        grams = get_grams(lines)
        save_picklized(grams, grams_filename)
        return grams


def build_evaluate():
    dataset = 'heldout'
    lines = load_gt(tokenize=False, dataset=dataset)
    gt_grams = get_gt_grams_cached(lines, dataset)
    evaluate(FLAGS.INPUT_SAMPLE, gt_grams)


if __name__ == '__main__':
    method = 'train'
    if method == 'train':
        build_run()
    if method == 'test':
        generate()
    if method == 'evaluate':
        build_evaluate()