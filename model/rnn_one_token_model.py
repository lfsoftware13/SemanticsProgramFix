import more_itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz.sandbox import unzip

from common.logger import info
from common.problem_util import to_cuda
from common.torch_util import pad_one_dim_of_tensor_list, create_sequence_length_mask, expand_tensor_sequence_to_same
from common.util import PaddedList
from model.graph_encoder import GraphEncoder
from model.seq2seq import EncoderRNN, DecoderRNN
from model.seq2seq.attention import Attention


class SelfPointerNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(SelfPointerNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.transform = nn.Linear(hidden_size, hidden_size)
        self.query_vector = nn.Parameter(torch.rand(1, hidden_size, 1))
        self.query_transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_seq, query, mask=None):
        batch_size = input_seq.shape[0]
        input_seq = self.transform(input_seq)
        x = input_seq + self.query_transform(query)
        x = F.tanh(x)
        x = torch.bmm(x, self.query_vector.expand(batch_size, -1, -1)).view(batch_size, -1)
        x = torch.squeeze(x, -1)
        if mask is not None:
            shape_list = [1 for i in range(len(x.shape))]
            shape_list[0] = batch_size
            shape_list[1] = -1
            mask = mask.view(*shape_list)
            x = x.masked_fill_(~mask, value=float('-inf'))
        x = F.softmax(x, dim=-1)
        return x


class PositionPointerNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(PositionPointerNetwork, self).__init__()
        self.position_linear = nn.Linear(hidden_size, hidden_size)
        self.position_pointer = SelfPointerNetwork(hidden_size=hidden_size)
        self.position_query = nn.Parameter(torch.rand(1, 1, hidden_size))

    def forward(self, encoder_output, encoder_mask):
        encoder_output_state = self.position_linear(encoder_output)
        error_position = self.position_pointer(encoder_output_state, self.position_query, encoder_mask)
        return error_position


class LineRNNEncoderWrapper(nn.Module):
    def __init__(self, input_size, hidden_size, vocabulary_size, n_layers, max_length, input_dropout_p=0, dropout_p=0,
                 bidirectional=False, rnn_cell='GRU'):
        super(LineRNNEncoderWrapper, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional_num = 2 if bidirectional else 1

        self.line_encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=max_length, input_size=input_size,
                                       hidden_size=hidden_size,
                                       input_dropout_p=input_dropout_p, dropout_p=dropout_p,
                                       n_layers=n_layers,
                                       bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=False,
                                       embedding=None, update_embedding=True, do_embedding=False)
        self.line_encoder_hidden_linear = nn.Linear(n_layers * self.bidirectional_num * hidden_size,
                                                    hidden_size)

    def split_sequence_accroding_chunk_list(self, sequence_tensor, chunk_list, dim=0, fill_value=0):
        '''
        split one sequence to multiply chunk accroding chunk list
        :param sequence_tensor: [..., dim, ...]
        :param chunk_list: a list of each chunk size
        :return: [..., len(chunk_list), max(chunk_list), ...]
        '''
        if dim < 0:
            input_size_len = len(sequence_tensor.shape)
            dim = dim + input_size_len
        one_line_seq = torch.split(sequence_tensor, chunk_list, dim=dim)
        padded_one_line_seq = pad_one_dim_of_tensor_list(one_line_seq, dim=dim, fill_value=fill_value)
        line_seq = torch.stack(padded_one_line_seq, dim=dim)
        return line_seq

    def merge_sequence_accroding_chunk_list(self, line_tensor: torch.Tensor, chunk_list, dim=0):
        '''
        merge multiply chunk to one sequence accroding chunk list
        :param line_tensor: [..., len(chunk_list), max(chunk_list), ...]
        :param chunk_list: a list of each chunk size
        :return: [..., sequence, ...]
        '''
        line_shape = list(line_tensor.shape)
        new_shape = line_shape[:dim] + [-1] +line_shape[dim+2:]
        line_tensor = line_tensor.contiguous().view(*new_shape)

        chunk_index_mask = create_sequence_length_mask(chunk_list).view(-1)
        index_shape = [-1 for _ in new_shape]
        index_shape[dim] = chunk_index_mask.shape[0]
        chunk_index_mask = chunk_index_mask.view(*index_shape)
        line_tensor = torch.masked_select(line_tensor, mask=chunk_index_mask).view(*new_shape)
        return line_tensor

    def do_line_encoder(self, input_sequence, line_list):
        '''

        :param input_sequence:
        :return:
        '''
        output, hidden = self.line_encoder(input_sequence, line_list)
        return output, hidden

    def forward(self, input_seq, input_token_length):
        '''

        :param input_seq:
        :param input_token_length:
        :return:
        batch_token_sequence: [batch_size, tokens, hidden_size]
        batch_line_sequence: [batch_size, lines, hidden_size]
        batch_line_hidden: [num_layers * bi_num, batch_size, line_num, hidden_size]
        '''
        batch_size = input_seq.shape[0]
        input_seq_batch_list = torch.unbind(input_seq)
        line_length_list = torch.unbind(input_token_length)
        line_hidden_list = []
        token_hidden_list = []
        for i in range(batch_size):
            one_input_seq = input_seq_batch_list[i]
            line_list = line_length_list[i]
        # for one_input_seq, line_list in zip(input_seq_batch_list, line_length_list):
            # split sequence to padded line sequence
            one_input_seq = one_input_seq[:torch.sum(line_list).tolist()]
            line_seq = self.split_sequence_accroding_chunk_list(one_input_seq, line_list.tolist(), dim=0)
            token_output, line_hidden = self.do_line_encoder(line_seq, line_list)
            line_hidden_list += [line_hidden]
            token_hidden_list += [self.merge_sequence_accroding_chunk_list(token_output, line_list)]

        padded_line_hidden_list = pad_one_dim_of_tensor_list(line_hidden_list, dim=0, fill_value=0)
        batch_line_hidden = torch.stack(padded_line_hidden_list, dim=1)
        padded_token_hidden_list = pad_one_dim_of_tensor_list(token_hidden_list, dim=0, fill_value=0)
        batch_token_sequence = torch.stack(padded_token_hidden_list, dim=0)

        # batch_line_hidden: [num_layers * bi_num, batch_size, lines, hidden_size]
        line_num = batch_line_hidden.shape[2]
        batch_line_sequence = batch_line_hidden.permute((1, 2, 0, 3)).contiguous().view((batch_size, line_num, -1)).contiguous()
        batch_line_sequence = self.line_encoder_hidden_linear(batch_line_sequence)

        # batch_token_sequence: [batch_size, tokens, hidden_size]
        # batch_line_sequence: [batch_size, lines, hidden_size]
        # batch_line_hidden: [num_layers * bi_num, batch_size, line_num, hidden_size]
        return batch_token_sequence, batch_line_sequence, batch_line_hidden

class LineRNNModel(nn.Module):

    def __init__(self, vocabulary_size, input_size, hidden_size, encoder_layer_nums, decoder_layer_nums, max_length,
                 max_sample_length, begin_token, end_token, input_dropout_p=0, dropout_p=0, bidirectional=True,
                 rnn_cell='GRU', use_attention=True, graph_embedding=None, graph_parameter={}):
        super(LineRNNModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder_layer_nums = encoder_layer_nums
        self.max_length = max_length
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.bidirectional_num = 2 if bidirectional else 1
        self.rnn_cell = rnn_cell
        self.graph_embedding = graph_embedding
        self.graph_parameter = graph_parameter

        self.embedding = nn.Embedding(vocabulary_size, input_size)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if graph_embedding is not None:
            self.graph_encoder = GraphEncoder(hidden_size, graph_embedding=graph_embedding, graph_parameter=graph_parameter)

        self.line_encoder = LineRNNEncoderWrapper(input_size=input_size, hidden_size=hidden_size,
                                                  vocabulary_size=vocabulary_size, n_layers=encoder_layer_nums,
                                                  max_length=max_length, input_dropout_p=input_dropout_p,
                                                  dropout_p=dropout_p, bidirectional=bidirectional, rnn_cell=rnn_cell)
        self.code_encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=max_length, input_size=input_size,
                                       hidden_size=hidden_size, input_dropout_p=input_dropout_p, dropout_p=dropout_p,
                                       n_layers=encoder_layer_nums, bidirectional=bidirectional, rnn_cell=rnn_cell,
                                       variable_lengths=False, embedding=None, update_embedding=True, do_embedding=False)
        self.encoder_linear = nn.Linear(hidden_size * self.bidirectional_num, hidden_size//2)

        self.position_pointer = PositionPointerNetwork(hidden_size=self.bidirectional_num * hidden_size)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=max_sample_length,
                                  hidden_size=hidden_size,
                                  sos_id=begin_token, eos_id=end_token, n_layers=decoder_layer_nums, rnn_cell=rnn_cell,
                                  bidirectional=bidirectional, input_dropout_p=input_dropout_p, dropout_p=dropout_p,
                                  use_attention=use_attention)

    def forward(self, input_seq, input_line_length: torch.Tensor, input_line_token_length: torch.Tensor, input_length: torch.Tensor, adj_matrix,
                target_seq, target_length, do_sample=False):
        '''

        :param input_seq: input sequence, torch.Tensor, full with token id in dictionary
        :param input_line_length: the number of lines per input sequence, [batch]
        :param input_line_token_length: the length of each lines, [batch, line]
        :param input_length: input token length include ast node, [batch]
        :param adj_matrix:
        :param target_seq:
        :param target_length:
        :param do_sample:
        :return:
        '''
        teacher_forcing_ratio = 0 if do_sample else 1
        embedded = self.embedding(input_seq)
        embedded = self.input_dropout(embedded)

        if self.graph_embedding is not None:
            copy_length = torch.sum(input_line_token_length, dim=-1)
            graph_embedded = self.graph_encoder.forward(adjacent_matrix=adj_matrix, copy_length=copy_length,
                                                  input_seq=embedded)
        else:
            graph_embedded = embedded

        batch_token_sequence, batch_line_sequence, batch_line_hidden = self.line_encoder.forward(graph_embedded, input_line_token_length)
        # code_state: [num_layers* bi_num, batch_size, hidden_size]
        line_output_state, code_state = self.code_encoder(batch_line_sequence, input_line_length)

        line_mask = create_sequence_length_mask(input_line_length)
        error_position = self.position_pointer(line_output_state, line_mask)
        pos = torch.max(error_position, dim=-1)[1]

        error_line_hidden_list = [batch_line_hidden[:, i, p] for i, p in enumerate(pos)]
        error_line_hidden = torch.stack(tuple(error_line_hidden_list), dim=1)
        # error_line_hidden = batch_line_hidden[:, :, pos]
        combine_line_state = self.encoder_linear(torch.cat((error_line_hidden, code_state), dim=-1))

        encoder_mask = create_sequence_length_mask(input_length, max_len=graph_embedded.shape[1])
        # combine_line_state: [num_layers, batch, hidden_size]
        # batch_token_sequence: [batch, tokens, hidden_size]
        decoder_outputs, decoder_hidden, _ = self.decoder(inputs=target_seq, encoder_hidden=combine_line_state, encoder_outputs=graph_embedded,
                    function=F.log_softmax, teacher_forcing_ratio=teacher_forcing_ratio, encoder_mask=~encoder_mask)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return error_position, decoder_outputs


def create_loss_fn(ignore_id):
    cross_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def loss_fn(error_position, output_tokens, target_position, target_tokens):
        # target_tokens = target_tokens[:, 1:]
        output_loss = cross_loss_fn(output_tokens.permute(0, 2, 1), target_tokens)
        position_loss = cross_loss_fn(error_position, target_position)
        total_loss = output_loss + position_loss
        return total_loss
    return loss_fn


def create_parse_input_batch_data_fn(ignore_id, use_ast=False):
    def parse_input(batch_data, do_sample=False):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['error_token_ids'], fill_value=0)))
        input_line_length = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line_length'])))
        input_line_token_length = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line_token_length'])))

        input_length = to_cuda(torch.LongTensor(PaddedList(batch_data['error_token_length'])))
        if not use_ast:
            adj_matrix = to_cuda(torch.LongTensor(batch_data['adj']))
        else:
            adjacent_tuple = [[[i] + tt for tt in t] for i, t in enumerate(batch_data['adj'])]
            adjacent_tuple = [list(t) for t in unzip(more_itertools.flatten(adjacent_tuple))]
            size = max(batch_data['error_token_length'])
            # print("max length in this batch:{}".format(size))
            adjacent_tuple = torch.LongTensor(adjacent_tuple)
            adjacent_values = torch.ones(adjacent_tuple.shape[1]).long()
            adjacent_size = torch.Size([len(batch_data['error_token_length']), size, size])
            info('batch_data input_length: ' + str(batch_data['error_token_length']))
            info('size: ' + str(size))
            info('adjacent_tuple: ' + str(adjacent_tuple.shape))
            info('adjacent_size: ' + str(adjacent_size))
            adj_matrix = to_cuda(
                torch.sparse.LongTensor(
                    adjacent_tuple,
                    adjacent_values,
                    adjacent_size,
                ).float().to_dense()
            )

        if not do_sample:
            target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_ids'], fill_value=ignore_id)))
            target_length = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_length'])))
        else:
            target_seq = None
            target_length = None

        return input_seq, input_line_length, input_line_token_length, input_length, adj_matrix, target_seq, target_length
    return parse_input


def create_parse_target_batch_data_fn(ignore_id, no_target=False):
    def parse_target(batch_data):
        if 'error_line' not in batch_data.keys() or no_target:
            return None
        target_error_position = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line'])))
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_ids'], fill_value=ignore_id)))
        target_seq = target_seq[:, 1:]
        return target_error_position, target_seq
    return parse_target


def expand_output_and_target_fn(ignore_id):
    def expand_fn(model_output, model_target):
        error_position, decoder_outputs = model_output
        target_error_position, target_seq = model_target

        decoder_outputs, target_seq = expand_tensor_sequence_to_same(decoder_outputs, target_seq, fill_value=ignore_id)
        return (error_position, decoder_outputs), (target_error_position, target_seq)
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
        position_o, token_ids_o = model_output
        outputs = torch.squeeze(torch.topk(F.softmax(token_ids_o, dim=-1), dim=-1, k=1)[1], dim=-1)
        return outputs
    return create_output_ids


def print_output_fn(vocabulary, eos_id):
    def print_fn(final_output, model_output, model_target, model_input, batch_data, step_i):
        position_o = model_output[0]
        position = torch.squeeze(torch.topk(F.softmax(position_o, dim=-1), dim=-1, k=1)[1], dim=-1)
        position_t = model_target[0]
        token_ids = model_target[1]
        for i, (p, o, pt, ot) in enumerate(zip(position, final_output, position_t, token_ids)):
            o = o.tolist()
            ot = ot.tolist()
            try:
                end_pos = ot.index(eos_id)
                end_pos += 1
            except ValueError as e:
                end_pos = len(o)
            o = o[:end_pos]
            texts = [vocabulary.id_to_word(t) for t in o]
            tests_target = [vocabulary.id_to_word(t) for t in ot][:end_pos]

            info('in step {} iter {}'.format(step_i, i))
            info('position: {}'.format(p))
            info('position: {}'.format(pt))
            info('output: {}'.format(texts))
            info('target: {}'.format(tests_target))
    return print_fn





