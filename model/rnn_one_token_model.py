import torch
import torch.nn as nn
import torch.nn.functional as F

from common.torch_util import pad_one_dim_of_tensor_list, create_sequence_length_mask
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
        x = torch.bmm(x, self.query_vector.view(batch_size, -1, -1)).view(batch_size, -1)
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
        self.position_query = nn.Parameter(torch.rand(1, hidden_size, 1))

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


class RNNOneTokenModel(nn.Module):

    def __init__(self, vocabulary_size, input_size, hidden_size, encoder_layer_nums, decoder_layer_nums, max_length,
                 begin_token, end_token, input_dropout_p=0, dropout_p=0, bidirectional=True, rnn_cell='GRU'):
        super(RNNOneTokenModel, self).__init__()
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
        self.position_pointer = PositionPointerNetwork(hidden_size=hidden_size)
        self.decoder = DecoderRNN(vocab_size=vocabulary_size, max_len=max_length, hidden_size=hidden_size,
                                      sos_id=begin_token, eos_id=end_token, n_layers=decoder_layer_nums, rnn_cell=rnn_cell,
                                      bidirectional=False, input_dropout_p=input_dropout_p, dropout_p=dropout_p,
                                      use_attention=True)


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
        line_tensor = line_tensor.view(*new_shape)

        chunk_index_tensor = create_sequence_length_mask(chunk_list)
        chunk_index_tensor = chunk_index_tensor.view(-1)
        line_tensor = torch.index_select(line_tensor, dim=dim, index=chunk_index_tensor)
        return line_tensor

    def do_line_encoder(self, input_sequence, line_list):
        '''

        :param input_sequence:
        :return:
        '''
        output, hidden = self.line_encoder(input_sequence, line_list)
        return output, hidden

    def encode_lines(self, input_seq, input_token_length):
        input_seq_batch_list = torch.unbind(input_seq)
        line_length_list = torch.unbind(input_token_length)
        line_hidden_list = []
        token_hidden_list = []
        for one_input_seq, line_list in zip(input_seq_batch_list, line_length_list):
            # split sequence to padded line sequence
            line_seq = self.split_sequence_accroding_chunk_list(one_input_seq, line_list.tolist(), dim=0)
            token_output, line_hidden = self.do_line_encoder(line_seq, line_list)
            line_hidden_list += line_hidden
            token_hidden_list += self.merge_sequence_accroding_chunk_list(token_output, line_list.tolist())

        padded_line_hidden_list = pad_one_dim_of_tensor_list(line_hidden_list, dim=0, fill_value=0)
        batch_line_sequence = torch.stack(padded_line_hidden_list, dim=0)
        padded_token_hidden_list = pad_one_dim_of_tensor_list(token_hidden_list, dim=0, fill_value=0)
        batch_token_sequence = torch.stack(padded_token_hidden_list, dim=0)
        return batch_token_sequence, batch_line_sequence

    def forward(self, input_seq, input_line_length: torch.Tensor, input_line_token_length: torch.Tensor,
                target_seq, target_length):
        embedded = self.embedding(input_seq)
        embedded = self.input_dropout(embedded)

        batch_token_sequence, batch_line_sequence = self.encode_lines(embedded, input_line_token_length)
        line_output_state, code_state = self.code_encoder(batch_line_sequence, input_line_length)

        line_mask = create_sequence_length_mask(input_line_length)
        error_position = self.position_pointer(line_output_state, line_mask)

        token_length = torch.sum(input_line_token_length, dim=-1)
        token_mask = create_sequence_length_mask(token_length)
        decoder_outputs, decoder_hidden, _ = self.decoder(inputs=target_seq, encoder_hidden=code_state, encoder_outputs=line_output_state,
                    function=F.log_softmax, teacher_forcing_ratio=1, encoder_mask=token_mask)
        return error_position, decoder_outputs








