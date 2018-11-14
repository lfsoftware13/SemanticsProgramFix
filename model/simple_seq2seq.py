import torch
import torch.nn as nn
import torch.nn.functional as F

from common.torch_util import create_sequence_length_mask
from common.util import PaddedList
from common.problem_util import to_cuda
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
        self.bidirectional_num = 2 if bidirectional else 1
        self.use_attention = use_attention

        self.encoder = EncoderRNN(vocab_size=vocab_size, max_len=max_len, input_size=input_size, hidden_size=hidden_size,
                             input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=n_layers,
                             bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=False, embedding=None,
                             update_embedding=True)
        self.decoder = DecoderRNN(vocab_size=vocab_size, max_len=max_len, hidden_size=self.bidirectional_num * hidden_size, sos_id=begin_token,
                             eos_id=end_token, n_layers=n_layers, rnn_cell=rnn_cell, bidirectional=bidirectional,
                             input_dropout_p=input_dropout_p, dropout_p=dropout_p, use_attention=use_attention)
        self.seq_model = Seq2seq(encoder=self.encoder, decoder=self.decoder, decode_function=F.log_softmax)

    def forward(self, inputs, input_length, targets, target_length, do_sample=False):
        teacher_forcing_ratio = 0 if do_sample else 1
        result = self.seq_model(input_variable=inputs, input_lengths=input_length, target_variable=targets,
                                teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = torch.stack(result[0], dim=1)
        decoder_mask = create_sequence_length_mask(target_length - 1)
        # outputs = outputs.data.masked_fill_(~decoder_mask.unsqueeze(dim=2), 0)
        return [outputs]


def create_loss_fn(ignore_id):
    cross_loss = nn.CrossEntropyLoss(ignore_index=ignore_id)
    def loss_fn(output_tokens_o, target_tokens):
        loss = cross_loss(output_tokens_o.permute(0, 2, 1), target_tokens)
        # loss = cross_loss(output_tokens_o.view(-1, output_tokens_o.shape[-1]), target_tokens.view(-1))
        return loss
    return loss_fn


def create_parse_input_batch_data_fn(ignore_id):
    def parse_input(batch_data, do_sample=False):
        inputs = to_cuda(torch.LongTensor(PaddedList(batch_data['input_seq'])))
        input_length = to_cuda(torch.LongTensor(PaddedList(batch_data['input_length'])))
        if not do_sample:
            targets = to_cuda(torch.LongTensor(PaddedList(batch_data['target_seq'])))
            targets_length = to_cuda(torch.LongTensor(PaddedList(batch_data['target_length'])))
        else:
            targets = None
            targets_length = None
        return inputs, input_length, targets, targets_length
    return parse_input


def create_parse_target_batch_data_fn(ignore_id):
    def parse_output(batch_data):
        target_seq = [t[1:] for t in batch_data['target_seq']]
        targets = to_cuda(torch.LongTensor(PaddedList(target_seq, fill_value=ignore_id)))
        return [targets]
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


def extract_includes_fn():
    def extract_includes(input_data):
        return input_data['includes']
    return extract_includes


def create_output_ids_fn(end_id):
    def create_output_ids(model_output, model_input, do_sample=False):
        # outputs_o = model_output[0]
        # outputs = torch.squeeze(torch.topk(F.softmax(outputs_o, dim=-1), dim=-1, k=1)[1], dim=-1)
        return None, None
    return create_output_ids





