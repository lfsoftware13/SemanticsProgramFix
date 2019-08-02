import torch
import torch.nn as nn
import torch.nn.functional as F
from common import torch_util
from common.graph_embedding import GGNNLayer
from common.torch_util import create_sequence_length_mask
from model.seq2seq import EncoderRNN


class RNNGraphWrapper(nn.Module):
    def __init__(self, hidden_size, parameter):
        super().__init__()
        self.encoder = EncoderRNN(hidden_size=hidden_size, **parameter)
        self.bi = 2 if parameter['bidirectional'] else 1
        self.transform_size = nn.Linear(self.bi * hidden_size, hidden_size)

    def forward(self, x, adj, copy_length):
        o, _ = self.encoder(x)
        o = self.transform_size(o)
        return o


class MixedRNNGraphWrapper(nn.Module):
    def __init__(self,
                 hidden_size,
                 rnn_parameter,
                 graph_type,
                 graph_itr,
                 dropout_p=0,
                 mask_ast_node_in_rnn=False,
                 ):
        super().__init__()
        self.rnn = nn.ModuleList([RNNGraphWrapper(hidden_size, rnn_parameter) for _ in range(graph_itr)])
        self.graph_itr = graph_itr
        self.dropout = nn.Dropout(dropout_p)
        self.mask_ast_node_in_rnn = mask_ast_node_in_rnn
        self.inner_graph_itr = 1
        if graph_type == 'ggnn':
            self.graph = GGNNLayer(hidden_size)

    def forward(self, x, adj, copy_length):
        if self.mask_ast_node_in_rnn:
            copy_length_mask = create_sequence_length_mask(copy_length, x.shape[1]).unsqueeze(-1)
            zero_fill = torch.zeros_like(x)
            for i in range(self.graph_itr):
                tx = torch.where(copy_length_mask, x, zero_fill)
                tx = tx + self.rnn[i](tx, adj, copy_length)
                x = torch.where(copy_length_mask, tx, x)
                x = self.dropout(x)
                # for _ in range(self.inner_graph_itr):
                x = x + self.graph(x, adj)
                if i < self.graph_itr - 1:
                    # pass
                    x = self.dropout(x)
        else:
            for i in range(self.graph_itr):
                x = x + self.rnn[i](x, adj, copy_length)
                x = self.dropout(x)
                x = x + self.graph(x, adj)
                if i < self.graph_itr - 1:
                    x = self.dropout(x)
        return x


class MultiIterationGraphWrapper(nn.Module):
    def __init__(self,
                 hidden_size,
                 graph_type,
                 graph_itr,
                 dropout_p=0,
                 ):
        super().__init__()
        self.graph_itr = graph_itr
        self.dropout = nn.Dropout(dropout_p)
        self.inner_graph_itr = 1
        if graph_type == 'ggnn':
            self.graph = GGNNLayer(hidden_size)

    def forward(self, x, adj, copy_length):
        for i in range(self.graph_itr):
            x = x + self.graph(x, adj)
            if i < self.graph_itr - 1:
                x = self.dropout(x)
        return x


class GraphEncoder(nn.Module):
    def __init__(self,
                 hidden_size=300,
                 graph_embedding='ggnn',
                 graph_parameter={},
                 ):
        """
        :param hidden_size: The hidden state size in the model
        :param graph_embedding: The graph propagate method, option:["ggnn", "graph_attention", "rnn"]
        :param pointer_type: The method to point out the begin and the end, option:["itr", "query"]
        """
        super().__init__()
        self.graph_embedding = graph_embedding
        if graph_embedding == 'ggnn':
            self.graph = MultiIterationGraphWrapper(hidden_size=hidden_size, **graph_parameter)
        elif graph_embedding == 'rnn':
            self.graph = RNNGraphWrapper(hidden_size=hidden_size, parameter=graph_parameter)
        elif graph_embedding == 'mixed':
            self.graph = MixedRNNGraphWrapper(hidden_size, **graph_parameter)

    def forward(self,
                adjacent_matrix,
                input_seq,
                copy_length,
                ):
        input_seq = self.graph(input_seq, adjacent_matrix, copy_length)
        return input_seq