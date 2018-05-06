from pyduyp.logger.log import log

from tensorflow.contrib import rnn

output_keep_prob = 1.0
input_keep_prob = 1.0
num_layers = 2
rnn_size = 5
training = True

cell_fn = rnn.BasicRNNCell
cells = []
for _ in range(num_layers):
    cell = cell_fn(rnn_size)
    if training and (output_keep_prob < 1.0 or input_keep_prob < 1.0):
        cell = rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
        log.info("cell:{}".format(cell))
    cells.append(cell)

print(cells)