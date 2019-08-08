import json

from common.util import create_python_tokenize_fn, filter_python_special_token


def create_name_list_by_LexToken(code_obj_list):
    name_list = [''.join(obj.value) if isinstance(obj.value, list) else obj.value for obj in code_obj_list]
    return name_list


def parse_xy_sequence(df, data_type, keyword_vocab, tokenize_fn=None, add_begin_end_label=True):
    df['code_tokens'] = df['code'].map(tokenize_fn)
    df = df[df['code_tokens'].map(lambda x: x is not None)].copy()
    df['code_tokens'] = df['code_tokens'].map(list)
    print('after tokenize: ', len(df.index))

    transform_word_to_id_fn = lambda name_list: keyword_vocab.parse_text([name_list], False)[0]
    df['code_words'] = df['code_tokens'].map(create_name_list_by_LexToken)
    df['error_code_word_id'] = df['code_words'].map(transform_word_to_id_fn)

    df['similar_code_tokens'] = df['similar_code'].map(tokenize_fn)
    df = df[df['similar_code_tokens'].map(lambda x: x is not None)].copy()
    df['similar_code_tokens'] = df['similar_code_tokens'].map(list)
    print('after tokenize: ', len(df.index))

    df['similar_code_words'] = df['similar_code_tokens'].map(create_name_list_by_LexToken)
    df['similar_code_word_id'] = df['similar_code_words'].map(transform_word_to_id_fn)

    if add_begin_end_label:
        begin_id = keyword_vocab.word_to_id(keyword_vocab.begin_tokens[0])
        end_id = keyword_vocab.word_to_id(keyword_vocab.end_tokens[0])
        add_fn = lambda x: [begin_id] + x + [end_id]
        add_label_fn = lambda x: [keyword_vocab.begin_tokens[0]] + x + [keyword_vocab.end_tokens[0]]
        df['error_code_word_id'] = df['error_code_word_id'].map(add_fn)
        df['code_words'] = df['code_words'].map(add_label_fn)
        df['similar_code_word_id'] = df['similar_code_word_id'].map(add_fn)
        df['similar_code_words'] = df['similar_code_words'].map(add_label_fn)

    return df['error_code_word_id'], df['code_words'], df['similar_code_word_id'], df['similar_code_words']


def parse_simple_python_error_code(df, data_type, keyword_vocab, max_sample_length, add_begin_end_label=True, only_sample=False):
    tokenize_fn = create_python_tokenize_fn()
    get_token_str_fn = lambda x: [i[1] for i in x]
    get_token_line_fn = lambda x: [i[2][0] for i in x]

    transform_word_to_id_without_position = lambda name_list: keyword_vocab.parse_text([name_list], False)[0]
    df = df[df['artificial_code'].map(lambda x: x is not None and x.strip() != '')].copy()
    df['raw_error_tokens'] = df['artificial_code'].map(tokenize_fn)
    df['raw_error_tokens'] = df['raw_error_tokens'].map(filter_python_special_token)
    df['error_tokens'] = df['raw_error_tokens'].map(get_token_str_fn)
    df['error_token_ids'] = df['error_tokens'].map(transform_word_to_id_without_position)

    df['error_tokens_line'] = df['raw_error_tokens'].map(get_token_line_fn)
    df['error_line_token_length'] = df['error_tokens_line'].map(count_line_length)

    def transform_word_to_id(name_list):
        return keyword_vocab.parse_text([name_list], True)[0]

    if not only_sample:
        df['change_record'] = df['change_record'].map(json.loads)

        df['after_token'] = df['change_record'].map(lambda x: x['after'])
        df['after_token'] = df['after_token'].map(tokenize_fn)
        df['after_token'] = df['after_token'].map(filter_python_special_token)
        df = df[df['after_token'].map(lambda x: len(x) < max_sample_length)].copy()
        df['change_after_tokens'] = df['after_token'].map(get_token_str_fn)
        df['change_after_tokens_ids'] = df['change_after_tokens'].map(transform_word_to_id)

        df['original_token'] = df['change_record'].map(lambda x: x['original'])
        df['original_token'] = df['original_token'].map(tokenize_fn)
        df['original_token'] = df['original_token'].map(filter_python_special_token)
        df = df[df['original_token'].map(lambda x: len(x) < max_sample_length)].copy()
        df['change_original_tokens'] = df['original_token'].map(get_token_str_fn)
        df['change_original_tokens_ids'] = df['change_original_tokens'].map(transform_word_to_id)

        df['error_type'] = df['change_record'].map(lambda x: x['errorType'])
        df['error_line'] = df['change_record'].map(lambda x: x['row']-1)

    return df


def count_line_length(token_lines):
    now = token_lines[0]
    line_len_list = []
    line_len = 0
    for l in token_lines:
        if l != now:
            line_len_list.append(line_len)
            now = l
            line_len = 0
        line_len += 1
    line_len_list.append(line_len)
    return line_len_list
