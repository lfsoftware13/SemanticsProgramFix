

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
