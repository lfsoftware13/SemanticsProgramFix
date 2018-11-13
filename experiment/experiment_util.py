from common.pycparser_util import tokenize_by_clex_fn
from experiment.load_data_vocabulary import create_deepfix_common_error_vocabulary
from experiment.parse_xy_util import parse_xy_sequence
from read_data.read_experiment_data import read_fake_common_deepfix_error_dataset_with_limit_length


def load_fake_deepfix_dataset_iterate_error_data(is_debug=False):
    vocab = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                   end_tokens=['<END>', '<INNER_END>'], unk_token='<UNK>',
                                                   addition_tokens=['<PAD>'])

    train, valid, test = read_fake_common_deepfix_error_dataset_with_limit_length(500)

    if is_debug:
        train = train.sample(100)
        valid = valid.sample(100)
        test = test.sample(100)

    tokenize_fn = tokenize_by_clex_fn()
    parse_fn = parse_xy_sequence
    add_begin_end_label = True
    parse_param = [vocab, tokenize_fn, add_begin_end_label]

    train_data = parse_fn(train, 'train', *parse_param)
    valid_data = parse_fn(valid, 'valid', *parse_param)
    test_data = parse_fn(test, 'test', *parse_param)

    train = train.loc[train_data[0].index.values]
    valid = valid.loc[valid_data[0].index.values]
    test = test.loc[test_data[0].index.values]

    train_dict = {'error_token_id_list': train_data[0], 'error_token_name_list': train_data[1],
                  'target_token_id_list': train_data[2], 'target_token_name_list': train_data[3],
                  'includes': train['includes'], 'distance': train['distance'], 'id': train['id'], }
    valid_dict = {'error_token_id_list': valid_data[0], 'error_token_name_list': valid_data[1],
                  'target_token_id_list': valid_data[2], 'target_token_name_list': valid_data[3],
                  'includes': valid['includes'], 'distance': valid['distance'], 'id': valid['id'], }
    test_dict = {'error_token_id_list': test_data[0], 'error_token_name_list': test_data[1],
                  'target_token_id_list': test_data[2], 'target_token_name_list': test_data[3],
                  'includes': test['includes'], 'distance': test['distance'], 'id': test['id'], }

    return train_dict, valid_dict, test_dict


if __name__ == '__main__':
    dict_list = load_fake_deepfix_dataset_iterate_error_data(True)
    print(dict_list)
