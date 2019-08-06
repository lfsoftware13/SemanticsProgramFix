import parser
import symbol
import token
import keyword
from collections import defaultdict
import more_itertools

from common.util import create_python_tokenize_fn, filter_python_special_token


def load_parser_node_to_id_dict():
    id_to_node_dict = load_parser_id_to_node_dict()
    d = {v: k for k, v in id_to_node_dict.items()}
    return d


def load_parser_id_to_node_dict():
    d = {}
    d.update(token.tok_name)
    d.update(symbol.sym_name)
    return d


def load_all_parser_node_names():
    id_to_node_dict = load_parser_id_to_node_dict()
    l = id_to_node_dict.values()
    return l


# ---------------------- convert dict-tree to adjacent matrix ------------------------- #
def dict_tree_to_matrix(tree, n):
    return dtree_to_mat(tree, [[0 for i in range(n)] for j in range(n)])


def dtree_to_mat(tree, mat):
    for k, v in tree.items():
        for i in v.keys():
            mat[i][k] = mat[k][i] = 1
        mat = dtree_to_mat(v, mat)
    return mat


def dict_tree_to_tuple_list(tree):
    tuple_list = []
    for k, v in tree.items():
        for i in v.keys():
            node = create_link_node(k, i, 'ast_node')
            if node is not None:
                tuple_list.append(node)
        tuple_list += dict_tree_to_tuple_list(v)
    return tuple_list


# ---------------------- change a list-tree to dict-tree ------------------------------ #
def list2dict_tree(l):
    return {l[0]: list2dict_tree_helper(l[1:])}


def list2dict_tree_helper(l):
    d = {}
    for i in l:
        d.update(list2dict_tree(i))
    return d


# ----------------------------- parser list-tree util methods ------------------------------- #
test_nodes = [274, 305, 306, 309, 310, 311, 325, 331]
def construct_filter_list_tree(l, test_mode=False):
    if len(l) == 0:
        return l
    if token.ISTERMINAL(l[0]):
        return l
    current_id = l[0]
    current_test_mode = True if current_id in test_nodes or test_mode else False

    l_c = []
    for c in l[1:]:
        l_c += [construct_filter_list_tree(c, current_test_mode)]

    # remove current node(replace it by it child) if current node is (meet all conditions)
    # (1) under a test node
    # (2) has only one child
    # (3) the only child is a non-terminal node
    # (4) for special, the child is not the atom node(to keep the parent of the atom node)
    if current_test_mode \
            and len(l_c) == 1 \
            and token.ISNONTERMINAL(l_c[0][0]) \
            and l_c[0][0] != symbol.atom:
        return l_c[0]

    return [l[0]] + l_c


def print_list_tree(l, blank_nums=0):
    blanks = '  ' * blank_nums
    name = '' if token.ISNONTERMINAL(l[0]) else l[1]
    print('{}--{}  {}'.format(blanks, str(l[0]), name))
    if token.ISNONTERMINAL(l[0]):
        for c in l[1:]:
            print_list_tree(c, blank_nums + 1)


def list_tree_to_sequence(l, convert_to_str=False, walk_type='preorder'):
    """

    :param l: a list-tree
    :param walk_type: four type: {'preorder', 'inorder', 'postorder', 'breadth'}
    :return:
    """
    if walk_type == 'preorder':
        seq = list_tree_to_sequence_preorder(l)
    elif walk_type == 'inorder':
        raise NotImplementedError
    elif walk_type == 'postorder':
        raise NotImplementedError
    elif walk_type == 'breadth':
        raise NotImplementedError
    if convert_to_str:
        seq = [token.tok_name[s] if token.ISTERMINAL(s) else symbol.sym_name[s] for s in seq]
    return seq


def list_tree_to_sequence_preorder(l):
    seq = [l[0]]
    if token.ISTERMINAL(l[0]):
        return seq
    for c in l[1:]:
        seq += list_tree_to_sequence_preorder(c)
    return seq


# ------------------------------ index tree util ------------------------------ #
def replace_node_to_seq_index(l_tree, base_index=0, keep_terminal_info=False):
    """
    replace the node id in token.py and symbol.py to sequence index. For creating adjacent matrix, the node id should be
    replace by a unique number(we use index of the preorder sequence).
    :param l_tree:
    :param base_index: the first index number
    :param keep_terminal_info: keep the name, line info, col info or not in terminal nodes.
    :return:
    """

    seq_index = base_index

    def replace_list_preorder(l):
        nonlocal seq_index
        if token.ISTERMINAL(l[0]):
            ter_l = [seq_index]
            seq_index += 1
            if keep_terminal_info:
                ter_l += l[1:]
                ter_l += [l[0]]
            return ter_l

        index_list = []
        for i in l:
            if type(i) == int:
                index_list += [seq_index]
                seq_index += 1
            elif type(i) == list:
                index_list += [replace_list_preorder(i)]
        return index_list

    index_tree = replace_list_preorder(l_tree)
    return index_tree


def extract_terminal_node_info(l, replace_negative=False):
    line_no = l[2]
    col_no = 0 if l[3] < 0 and replace_negative else l[3]
    node_type = l[4]
    return l[0], l[1], line_no, col_no, node_type


file_only_nodes = {token.ENDMARKER}
line_only_nodes = {token.NEWLINE, token.INDENT, token.DEDENT}
def connect_one_terminal_node(l, token_seq, seq_i=0):
    if len(l) < 5:
        print('terminal node missing information. (line info or col info)')
        raise NotImplementedError
    original_seq_i = seq_i
    index_n, name_n, line_n, col_n, type_n = extract_terminal_node_info(l, replace_negative=True)

    while seq_i < len(token_seq):
        tok = token_seq[seq_i]
        token_type = tok[0]
        tok_name = tok[1]
        tok_start = tok[2]
        if type_n in file_only_nodes and type_n == token_type:
            return [l[0], [seq_i]], seq_i + 1
        elif type_n in line_only_nodes and type_n == token_type and line_n == tok_start[0]:
            return [l[0], [seq_i]], seq_i + 1
        elif tok_start == (line_n, col_n) and tok_name == name_n:
            return [l[0], [seq_i]], seq_i + 1
        seq_i += 1
    # can't find matching lextoken with current terminal node, reset the seq_index
    return [l[0]], original_seq_i


def add_lextoken_to_terminalnodes(token_seq, index_tree):
    """
    add lex tokens index to parse tree. Attention: the parse tree should have replaced the node id to sequence index
    :param token_seq:
    :param index_tree: parse tree whose node id have be replaced to sequence index
    :return:
    """
    seq_i = 0

    def search_tree_preorder(l):
        nonlocal seq_i
        if type(l) == int:
            return l
        if len(l) >= 1 and type(l[1]) == str:
            l, seq_i = connect_one_terminal_node(l, token_seq, seq_i=seq_i)
            return l
        else:
            l = [search_tree_preorder(i) for i in l]
            return l

    return search_tree_preorder(index_tree)


# ---------------------------- add special link for token sequence ------------------------------ #
def create_link_node(a, b, link_type="node_map"):
    if a is not None and b is not None:
        return (a, b, link_type)
    return None


def create_sequence_link(tokens):
    sequence_links = [create_link_node(i, i + 1, "seq_link") for i in range(len(tokens[:-1]))]
    sequence_links = list(filter(lambda x: x is not None, sequence_links))
    return sequence_links


def create_same_identifier_link(tokens):
    identifier_pos_map = defaultdict(list)
    for i, tok in enumerate(tokens):
        if tok[0] == token.NAME and not keyword.iskeyword(tok[1]):
            identifier_pos_map[tok[1]].append(i)

    identifiers_links = []
    for k, v in identifier_pos_map.items():
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                node = create_link_node(v[i], v[j], "same_name_link")
                if node is not None:
                    identifiers_links.append(node)
    return identifiers_links


def create_special_link_list(tokens, add_sequence_link=False):
    link_tuple_list = []
    if add_sequence_link:
        link_tuple_list += create_sequence_link(tokens)
    link_tuple_list += create_same_identifier_link(tokens)
    return link_tuple_list


# ---------------------------------------- ------------------------------------------ #
def load_python_parse_tree(code, filter_test=False, line_info=True, col_info=True):
    """
    input a python code, return the parse tree. the tree is represented in list-tree.
    :param code:
    :param filter_test: filter the redundant nodes in parse tree. The redundant nodes are introduced by the grammar.
    :param line_info:
    :param col_info:
    :return:
    """
    try:
        st_obj = parser.suite(code)
    except Exception as e:
        return None
    st_list = parser.st2list(st_obj, line_info=line_info, col_info=col_info)
    if filter_test:
        st_list = construct_filter_list_tree(st_list)
        # print_list_tree(st_list)
    return st_list


class ParseTreeGraph(object):
    def __init__(self, tokens, parse_tree, add_sequence_link=False):
        self.tokens = tokens
        self.original_parse_tree = parse_tree
        self.parse_tree = construct_filter_list_tree(self.original_parse_tree)
        self.tree_sequence = list_tree_to_sequence(self.parse_tree, convert_to_str=True, walk_type='preorder')


def generate_adjacent_matrix(st_list, tokens, delimiter_num=0, adjacent_type='adj'):
    index_list_tree = replace_node_to_seq_index(st_list, base_index=len(tokens) + delimiter_num, keep_terminal_info=True)
    full_index_list_tree = add_lextoken_to_terminalnodes(tokens, index_list_tree)
    full_index_dict_tree = list2dict_tree(full_index_list_tree)
    if adjacent_type == 'adj':
        adj = dict_tree_to_matrix(full_index_dict_tree, len(tokens) + delimiter_num + len(seq))
    elif adjacent_type == 'tuple':
        adj = dict_tree_to_tuple_list(full_index_dict_tree)
    else:
        adj = dict_tree_to_matrix(full_index_dict_tree, len(tokens) + delimiter_num + len(seq))
    return adj


def load_python_parse_graph(code, adjacent_type='adj'):
    tokenize_fn = create_python_tokenize_fn()
    tokens = tokenize_fn(code)
    tokens = filter_python_special_token(tokens)
    st_list = load_python_parse_tree(code, filter_test=True, line_info=True, col_info=True)

    token_sequence = [t[1] for t in tokens]
    delimiter_sequence = ["<Delimiter>"]
    tree_sequence = list_tree_to_sequence(st_list, convert_to_str=True, walk_type='preorder')
    input_str_sequence = token_sequence + delimiter_sequence + tree_sequence

    link_tuple_list = create_special_link_list(tokens, add_sequence_link=True)
    adj = generate_adjacent_matrix(st_list, tokens, delimiter_num=len(delimiter_sequence), adjacent_type=adjacent_type)

    if adjacent_type == 'adj':
        for l1, l2, _ in link_tuple_list:
            adj[l1][l2] = 1
            adj[l2][l1] = 1
    elif adjacent_type == 'tuple':
        adj += link_tuple_list
        adj = [[[l1, l2], [l2, l1]] for l1, l2, _ in adj]
        adj = list(more_itertools.flatten(adj))
    else:
        for l1, l2, _ in link_tuple_list:
            adj[l1][l2] = 1
            adj[l2][l1] = 1
    return st_list, input_str_sequence, adj


def parse_python_test():
    code = r'''
import os
def add(a:int, b):
    a = a + b
    if a > b and a > 1:
        pass
    return a
c = add(((1>2) and (2>3)), 2)
print(c)
            '''
    code2 = r'''def hello_world():
    return "hello world"'''
    st_obj = parser.suite(code)
    st_list = parser.st2list(st_obj, line_info=True, col_info=True)
    print(st_list)
    filtered_list = construct_filter_list_tree(st_list)
    print(filtered_list)
    print_list_tree(filtered_list)
    # root = construct_filter_tree(st_list)
    # root.print_tree(0)
    print('root')



if __name__ == '__main__':

    # parse_python_test()

    code = r'''
import os
def add(a:int, b):
    a = a + b
    if a > b and a > 1:
        pass
    return a
c = add(((1>2) and (2>3)), 2)
print(c)
                '''
    # st_list = load_python_parse_tree(code, filter_test=True)
    # print_list_tree(st_list)
    # d_tree = list2dict_tree(st_list)
    # print(d_tree)

    code2 = r'''def hello_world():
    a = 1 + 2
    if 2 > 1:
        return 3
    return "hello world"'''
    st_list = load_python_parse_tree(code2, filter_test=True)
    seq = list_tree_to_sequence(st_list, convert_to_str=False, walk_type='preorder')
    print(seq)

    tokenize_fn = create_python_tokenize_fn()
    tokens = tokenize_fn(code2)[1:]
    index_tree = replace_node_to_seq_index(st_list, base_index=len(tokens), keep_terminal_info=True)
    print(index_tree)
    full_index_tree = add_lextoken_to_terminalnodes(tokens, index_tree)
    print(full_index_tree)
    full_index_dict_tree = list2dict_tree(full_index_tree)
    print(full_index_dict_tree)
    tuple_list = dict_tree_to_tuple_list(full_index_dict_tree)
    adj = dict_tree_to_matrix(full_index_dict_tree, len(tokens) + len(seq))
    print(len(tuple_list), sum(more_itertools.collapse(adj)))
    # leaf_count = sum([1 if token.ISTERMINAL(x) else 0 for x in seq])
    # print('token length: {}, parse_tree_leaf_count: {}'.format(len(tokens), leaf_count))
