import json
from tokenize import tokenize
from io import BytesIO
import more_itertools

from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length, python_df_to_dataset

#创建python的token的字典
def python_token_dict():
    train_df, valid_df, test_df = python_df_to_dataset()
    
    train_set, valid_set, test_set = set(), set(), set()
    
    for index, row in train_df.iterrows():
        tokens = tokenize(BytesIO(row['code'].encode('utf-8')).readline)
        for token in tokens:
            train_set.add(token[1])

    for index, row in valid_df.iterrows():
        tokens = tokenize(BytesIO(row['code'].encode('utf-8')).readline)
        for token in tokens:
            valid_set.add(token[1])

    for index, row in test_df.iterrows():
        tokens = tokenize(BytesIO(row['code'].encode('utf-8')).readline)
        for token in tokens:
            test_set.add(token[1])
            
    train_dict, valid_dict, test_dict = dict(), dict(), dict()

    index = 0
    for i in train_set:
        train_dict[index] = i
        index += 1

    index = 0
    for i in valid_set:
        valid_dict[index] = i
        index += 1

    index = 0
    for i in test_set:
        test_dict[index] = i
        index += 1
    print('三种字典的大小', len(train_dict), len(valid_dict), len(test_dict))
    for i in test_dict:
        print(test_dict[i])
    return train_dict, valid_dict, test_dict

# deepfix fake error vocabulary
from vocabulary.word_vocabulary import load_vocabulary


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
    return ac_tokens | action_tokens


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
