import torch
import torch.nn as nn
import torch.nn.functional as F

from common.problem_util import to_cuda
from common.torch_util import pad_one_dim_of_tensor_list, create_sequence_length_mask
from common.util import PaddedList
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
        x = torch.bmm(x, self.query_vector.expand(batch_size, -1, -1)).view(batch_size, -1)
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


class OneTokenOutput(nn.Module):
    def __init__(self, hidden_size):
        super(OneTokenOutput, self).__init__()
        self.hidden_size = hidden_size


    def forward(self, hidden):
        pass


class OneTokenDecoderModel(nn.Module):
    def __init__(self):
        super(OneTokenDecoderModel, self).__init__()

    def forward(self, inputs):
        pass


class LineRNNModel(nn.Module):

    def __init__(self, vocabulary_size, input_size, hidden_size, encoder_layer_nums, decoder_layer_nums, max_length,
                 begin_token, end_token, input_dropout_p=0, dropout_p=0, bidirectional=True, rnn_cell='GRU',
                 use_attention=True):
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

        self.embedding = nn.Embedding(vocabulary_size, input_size)
        self.input_dropout = nn.Dropout(input_dropout_p)

        self.line_encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=max_length, input_size=input_size,
                                  hidden_size=hidden_size,
                                  input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=encoder_layer_nums,
                                  bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=False,
                                  embedding=None, update_embedding=True, do_embedding=False)
        self.code_encoder = EncoderRNN(vocab_size=vocabulary_size, max_len=max_length, input_size=input_size,
                                  hidden_size=hidden_size,
                                  input_dropout_p=input_dropout_p, dropout_p=dropout_p, n_layers=encoder_layer_nums,
                                  bidirectional=bidirectional, rnn_cell=rnn_cell, variable_lengths=False,
                                  embedding=None, update_embedding=True, do_embedding=False)
        self.line_encoder_hidden_linear = nn.Linear(encoder_layer_nums * self.bidirectional_num * hidden_size, hidden_size)
        self.encoder_linear = nn.Linear(hidden_size * self.bidirectional_num, hidden_size)
        self.position_pointer = PositionPointerNetwork(hidden_size=self.bidirectional_num * hidden_size)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=max_length,
                                  hidden_size=self.bidirectional_num * hidden_size,
                                  sos_id=begin_token, eos_id=end_token, n_layers=decoder_layer_nums, rnn_cell=rnn_cell,
                                  bidirectional=bidirectional, input_dropout_p=input_dropout_p, dropout_p=dropout_p,
                                  use_attention=use_attention)

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

    def encode_lines(self, input_seq, input_token_length):
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

    def forward(self, input_seq, input_line_length: torch.Tensor, input_line_token_length: torch.Tensor,
                target_seq, target_length):
        embedded = self.embedding(input_seq)
        embedded = self.input_dropout(embedded)

        batch_token_sequence, batch_line_sequence, batch_line_hidden = self.encode_lines(embedded, input_line_token_length)
        # code_state: [num_layers* bi_num, batch_size, hidden_size]
        line_output_state, code_state = self.code_encoder(batch_line_sequence, input_line_length)

        line_mask = create_sequence_length_mask(input_line_length)
        error_position = self.position_pointer(line_output_state, line_mask)

        pos = torch.max(error_position, dim=-1)[1]
        error_line_hidden_list = [batch_line_hidden[:, i, p] for i, p in enumerate(pos)]
        error_line_hidden = torch.stack(error_line_hidden_list, dim=1)
        # error_line_hidden = batch_line_hidden[:, :, pos]
        combine_line_state = self.encoder_linear(torch.cat((error_line_hidden, code_state), dim=-1))

        token_length = torch.sum(input_line_token_length, dim=-1)
        token_mask = create_sequence_length_mask(token_length)
        # combine_line_state: [num_layers, batch, hidden_size]
        # batch_token_sequence: [batch, tokens, hidden_size]
        decoder_outputs, decoder_hidden, _ = self.decoder(inputs=target_seq, encoder_hidden=combine_line_state, encoder_outputs=batch_token_sequence,
                    function=F.log_softmax, teacher_forcing_ratio=1, encoder_mask=~token_mask)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        return error_position, decoder_outputs


def create_loss_fn(ignore_id):
    cross_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_id)

    def loss_fn(error_position, output_tokens, target_position, target_tokens):
        target_tokens = target_tokens[:, 1:]
        output_loss = cross_loss_fn(output_tokens.permute(0, 2, 1), target_tokens)
        position_loss = cross_loss_fn(error_position, target_position)
        total_loss = output_loss + position_loss
        return total_loss
    return loss_fn


def create_parse_input_batch_data_fn(ignore_id):
    def parse_input(batch_data, do_sample=False):
        input_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['error_token_ids'], fill_value=0)))
        input_line_length = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line_length'])))
        input_line_token_length = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line_token_length'])))

        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_ids'], fill_value=ignore_id)))
        target_length = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_length'])))

        return input_seq, input_line_length, input_line_token_length, target_seq, target_length
    return parse_input


def create_parse_target_batch_data_fn(ignore_id):
    def parse_target(batch_data):
        target_error_position = to_cuda(torch.LongTensor(PaddedList(batch_data['error_line'])))
        target_seq = to_cuda(torch.LongTensor(PaddedList(batch_data['target_line_ids'], fill_value=ignore_id)))
        return target_error_position, target_seq
    return parse_target


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





