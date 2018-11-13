import torch
import torch.nn as nn
import torch.nn.functional as F

from .seq2seq.seq2seq import Seq2seq
from .seq2seq.EncoderRNN import EncoderRNN
from .seq2seq.DecoderRNN import DecoderRNN


class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, max_len, input_size, hidden_size, begin_token, end_token, input_dropout_p=0, dropout_p=0, n_layers=1,
                 bidirectional=False, rnn_cell='gru', use_attention=True):
        super(SimpleSeq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.begin_token = begin_token
        self.end_token = end_token
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        self.encoder = EncoderRNN(vocab_size=vocab_size, max_len=max_len, input_size=input_size, hidden_size=hidden_size,
                             input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers,
                             bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=False, embedding=None,
                             update_embedding=True)
        self.decoder = DecoderRNN(vocab_size=vocab_size, max_len=max_len, hidden_size=hidden_size, sos_id=begin_token,
                             eos_id=end_token, n_layers=n_layers, rnn_cell=rnn_cell, bidirectional=bidirectional,
                             input_dropout_p=input_dropout_p, dropout_p=dropout_p, use_attention=use_attention)
        self.seq_model = Seq2seq(encoder=self.encoder, decoder=self.decoder, decode_function=F.log_softmax)

    def forward(self, inputs, input_length, targets, do_sample):
        teacher_forcing_ratio = 0 if do_sample else 1
        result = self.seq_model(input_variable=inputs, input_lengths=input_length, target_variable=targets,
                                teacher_forcing_ratio=teacher_forcing_ratio)
        return result[0]


def create_loss_fn(ignore_id):
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
    def loss_fn(output_tokens_o, target_tokens):
        loss = cross_loss(output_tokens_o, target_tokens)
        return loss
    return loss_fn


def create_parse_input_batch_data_fn():
    def parse_input(batch_data, do_sample):
        pass
    return parse_input


def create_parse_target_batch_data_fn(ignore_id):
    def parse_output(batch_data, do_sample):
        pass
    return parse_output


def expand_output_and_target_fn(ignore_id):
    def expand_fn(model_output, model_target):
        pass
    return expand_fn


def multi_step_print_output_records_fn(begin_id, end_id, vocabulary=None):
    def multi_step_print_output(output_records, final_output, batch_data,
                                    step_i, compile_result_list):
        pass
    return multi_step_print_output




