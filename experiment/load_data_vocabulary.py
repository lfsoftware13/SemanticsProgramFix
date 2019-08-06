import json
from tokenize import tokenize
from io import BytesIO
import more_itertools

from common.constants import CACHE_DATA_PATH, pre_defined_py_label
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache, create_python_tokenize_fn
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length, \
    python_df_to_dataset, read_fake_semantic_python_dataset
from vocabulary.word_vocabulary import load_vocabulary


# 创建python的token的字典
def python_token_dict():
    train_df, _, _ = python_df_to_dataset()

    tokenize_fn = create_python_tokenize_fn()
    get_token_str_fn = lambda x: [i[1] for i in x]

    train_df['raw_tokens'] = train_df['code'].map(tokenize_fn)
    train_df['tokens'] = train_df['raw_tokens'].map(get_token_str_fn)

    tokens_list = train_df['tokens'].tolist()
    tokens_set = set(more_itertools.collapse(tokens_list))
            
    tokens_dict = dict()

    index = 0
    for i in tokens_set:
        tokens_dict[index] = i
        index += 1

    return tokens_dict


def extract_python_tokens_set(l):
    tokenize_fn = create_python_tokenize_fn()
    get_token_str_fn = lambda x: [i[1] for i in x]

    raw_tokens = l.map(tokenize_fn)
    tokens = raw_tokens.map(get_token_str_fn)

    tokens_list = tokens.tolist()
    tokens_set = set(more_itertools.collapse(tokens_list))
    return tokens_set


# ------------------------------- old fake python semantic data vocabulary ----------------------------- #

def extract_artificial_change_after_value(change_record):
    after_fn = lambda one: one['after']
    change_record = change_record.map(json.loads)
    after_lines = change_record.map(after_fn)
    return after_lines


def get_deepfix_python_semantics_train_ac_tokens():
    train_df, _, _ = python_df_to_dataset()
    tokens_set = extract_python_tokens_set(train_df['code'])
    return tokens_set


def get_deepfix_python_semantics_train_action_tokens():
    train_df, _, _ = python_df_to_dataset()
    after_lines = extract_artificial_change_after_value(train_df['change_record'])
    after_tokens_set = extract_python_tokens_set(after_lines)
    return after_tokens_set


@disk_cache(basename='get_deepfix_python_semantics_train_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_deepfix_python_semantics_train_token_vocabulary_set():
    ac_tokens_set = get_deepfix_python_semantics_train_ac_tokens()
    action_tokens_set = get_deepfix_python_semantics_train_action_tokens()
    return ac_tokens_set | action_tokens_set


@disk_cache(basename='get_deepfix_python_semantics_train_token_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_deepfix_python_semantics_train_token_vocabulary_id_map():
    word_list = sorted(get_deepfix_python_semantics_train_token_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_deepfix_python_semantics_common_error_vocabulary', directory=CACHE_DATA_PATH)
def create_deepfix_python_semantics_common_error_vocabulary(begin_tokens, end_tokens, unk_token, addition_tokens=None):
    vocab = load_vocabulary(load_vocabulary_fn=get_deepfix_python_semantics_train_token_vocabulary_set,
                            load_vocabulary_id_dict=get_deepfix_python_semantics_train_token_vocabulary_id_map,
                            begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                            addition_tokens=addition_tokens)
    return vocab

# ------------------------------- fake python semantic data vocabulary ----------------------------- #

def get_fake_python_semantics_train_ac_tokens():
    train_df, _, _ = read_fake_semantic_python_dataset()
    tokens_set = extract_python_tokens_set(train_df['code'])
    return tokens_set


def get_fake_python_semantics_train_action_tokens():
    train_df, _, _ = read_fake_semantic_python_dataset()
    after_lines = extract_artificial_change_after_value(train_df['change_record'])
    after_tokens_set = extract_python_tokens_set(after_lines)
    return after_tokens_set


@disk_cache(basename='get_fake_python_semantics_train_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_fake_python_semantics_train_token_vocabulary_set():
    ac_tokens_set = get_fake_python_semantics_train_ac_tokens()
    action_tokens_set = get_fake_python_semantics_train_action_tokens()
    return ac_tokens_set | action_tokens_set


@disk_cache(basename='get_fake_python_semantics_train_token_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_fake_python_semantics_train_token_vocabulary_id_map():
    word_list = sorted(get_fake_python_semantics_train_token_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_fake_python_semantics_common_error_vocabulary', directory=CACHE_DATA_PATH)
def create_fake_python_semantics_common_error_vocabulary(begin_tokens, end_tokens, unk_token, addition_tokens=None):
    vocab = load_vocabulary(load_vocabulary_fn=get_fake_python_semantics_train_token_vocabulary_set,
                            load_vocabulary_id_dict=get_fake_python_semantics_train_token_vocabulary_id_map,
                            begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                            addition_tokens=addition_tokens)
    return vocab


def load_python_ast_vocabulary(vocabulary):
    vocabulary.add_token("<Delimiter>")
    from common.python_ast_util import load_all_ast_nodes
    class_name_list = load_all_ast_nodes()
    for n in class_name_list:
        vocabulary.add_token(n)
    return vocabulary


def load_python_parser_vocabulary(vocabulary):
    vocabulary.add_token("<Delimiter>")
    from common.python_parse_util import load_all_parser_node_names
    node_name_list = load_all_parser_node_names()
    for n in node_name_list:
        vocabulary.add_token(n)
    return vocabulary


# --------------------------------------------- deepfix fake error vocabulary --------------------- #

# deepfix fake error vocabulary
def get_deepfix_train_ac_tokens_without_includes():
    train_df, _, _ = read_fake_common_deepfix_error_dataset_with_limit_length(500)
    transform_lextoken_to_token_fn = lambda token_list: [i.value for i in token_list]
    tokenize_fn = tokenize_by_clex_fn()
    parse_tokens = [transform_lextoken_to_token_fn(tokenize_fn(code)) for code in train_df['similar_code']]
    return parse_tokens


def read_deepfix_modify_action_token():
    train_df, _, _ = read_fake_common_deepfix_error_dataset_with_limit_length(500)
    train_df['modify_action_list'] = train_df['modify_action_list'].map(json.loads)
    extract_to_token_fn = lambda actions: [act['to_char'] for act in actions]
    act_tokens = [extract_to_token_fn(actions) for actions in train_df['modify_action_list']]
    return act_tokens


@disk_cache(basename='get_deepfix_train_token_vocabulary_set', directory=CACHE_DATA_PATH)
def get_deepfix_train_token_vocabulary_set():
    ac_parse_tokens = get_deepfix_train_ac_tokens_without_includes()
    # error_parse_tokens = get_deepfix_train_error_tokens_without_includes()
    action_tokens = read_deepfix_modify_action_token()

    ac_tokens = set(more_itertools.collapse(ac_parse_tokens))
    # err_tokens = set(more_itertools.collapse(error_parse_tokens))
    action_tokens = set(more_itertools.collapse(action_tokens))

    return ac_tokens | action_tokens | pre_defined_py_label


@disk_cache(basename='get_deepfix_train_token_vocabulary_id_map', directory=CACHE_DATA_PATH)
def get_deepfix_train_token_vocabulary_id_map():
    word_list = sorted(get_deepfix_train_token_vocabulary_set())
    return {word: i for i, word in enumerate(word_list)}


@disk_cache(basename='create_deepfix_common_error_vocabulary', directory=CACHE_DATA_PATH)
def create_deepfix_common_error_vocabulary(begin_tokens, end_tokens, unk_token, addition_tokens=None):
    vocab = load_vocabulary(get_deepfix_train_token_vocabulary_set, get_deepfix_train_token_vocabulary_id_map,
                            begin_tokens=begin_tokens, end_tokens=end_tokens, unk_token=unk_token,
                            addition_tokens=addition_tokens)
    return vocab


if __name__ == '__main__':
    vocab = create_fake_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                    unk_token='<UNK>', addition_tokens=['<PAD>'])
    print(vocab.vocabulary_size)
    vocab = load_python_parser_vocabulary(vocab)
    print(vocab.vocabulary_size)

