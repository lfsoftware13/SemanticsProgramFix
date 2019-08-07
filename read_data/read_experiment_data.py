import json

import pandas as pd
import os
from tokenize import tokenize
from io import BytesIO
from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache, filter_length, create_python_tokenize_fn
from read_data.collect_codeflaws_benchmark import read_and_pickle_codeflaws_data
from read_data.read_filter_data import read_fake_deepfix_common_error_records, filter_distinct_artificalCode, \
    python_filter, read_fake_semantic_python_records, read_codeforces_real_semantic_python_records


# 过滤代码token大于500的记录，并将一个df划分为三个数据集
@disk_cache(basename='python_df_to_dataset', directory=CACHE_DATA_PATH)
def python_df_to_dataset(max_length=500):
    df = python_filter()
    print(df.shape[0], '没过滤token的组数')

    # 过滤掉token大于500的记录
    for index, row in df.iterrows():
        tokens = tokenize(BytesIO(row['artificial_code'].encode('utf-8')).readline)
        if len(list(tokens)) > max_length:
            df.drop([index],inplace=True)
    df = df.reset_index(drop=True)
    print(df.shape[0], '过滤了token的组数')

    valid_df = df.sample(frac=0.1)
    df = df.drop(valid_df.index)
    test_df = df.sample(frac=0.1)
    train_df = df.drop(test_df.index)

    # 将每个数据集的索引重置为从0开始
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    print('三种数据集的数据个数', valid_df.shape[0], test_df.shape[0], train_df.shape[0])
    return train_df, valid_df, test_df


@disk_cache(basename='read_fake_semantic_python_dataset', directory=CACHE_DATA_PATH)
def read_fake_semantic_python_dataset(limit_length=500):

    valid_problem_ids = ['117091', '117992', '116707', '116708', '117993', '116347', '116709', '116346', '116350', '116712', '116348', '116225', '116224', '116711', '117994', '115519', '115518', '115517', '114248', '114249', '114247', '114243', '114244', '114242', '114250', '113852', '113851', '114251', '113846', '114548', '114546', '114545', '114544', '114547', '113854', '112698', '112699', '112700', '111649', '112027', '112028', '112029', '111651', '111650', '112250', '112247', '112248', '112246', '112245', '110089', '110088', '110364', '110087', '110365', '110366', '108813', '108816', '108814']
    test_problem_ids = ['110359', '108589', '108588', '108004', '108002', '108003', '107532', '107529', '106946', '107531', '106945', '106951', '106948', '106400', '106399', '107530', '106398', '106397', '106401', '105601', '105602', '105167', '105166', '106947', '106950', '105169', '105173', '105168', '105172', '104510', '104511', '104509', '104508', '104507', '104506', '107584', '107581', '107578', '107579', '107576', '107575', '107573', '103101', '107572', '103100', '105984', '105983', '103643', '103642', '105985', '103599', '103600', '105986', '103030', '103029', '103028', '102473', '102474', '102472', '106230', '106232', '103819', '106231', '103815', '103817', '100428', '100427', '100426', '103816', '100425', '98944']

    data_df = read_fake_semantic_python_records()
    tokenize_fn = create_python_tokenize_fn()
    data_df = filter_length(data_df, limit_length, tokenize_fn, code_key='code')
    print('after filter code length: {}'.format(len(data_df)))

    valid_df = data_df[data_df['problem_id'].map(lambda x: x in valid_problem_ids)]
    data_df = data_df.drop(valid_df.index)
    test_df = data_df[data_df['problem_id'].map(lambda x: x in test_problem_ids)]
    train_df = data_df.drop(test_df.index)

    # 将每个数据集的索引重置为从0开始
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    print('train data size: {}, valid data size: {}, test data size: {}'.format(train_df.shape[0], valid_df.shape[0], test_df.shape[0]))
    return train_df, valid_df, test_df


@disk_cache(basename='read_codeforces_real_semantic_python_dataset', directory=CACHE_DATA_PATH)
def read_codeforces_real_semantic_python_dataset(limit_length=500):

    test_problem_ids = ['110359', '108589', '108588', '108004', '108002', '108003', '107532', '107529', '106946', '107531', '106945', '106951', '106948', '106400', '106399', '107530', '106398', '106397', '106401', '105601', '105602', '105167', '105166', '106947', '106950', '105169', '105173', '105168', '105172', '104510', '104511', '104509', '104508', '104507', '104506', '107584', '107581', '107578', '107579', '107576', '107575', '107573', '103101', '107572', '103100', '105984', '105983', '103643', '103642', '105985', '103599', '103600', '105986', '103030', '103029', '103028', '102473', '102474', '102472', '106230', '106232', '103819', '106231', '103815', '103817', '100428', '100427', '100426', '103816', '100425', '98944']

    data_df = read_codeforces_real_semantic_python_records()
    tokenize_fn = create_python_tokenize_fn()
    data_df = filter_length(data_df, limit_length, tokenize_fn, code_key='code')
    print('after filter code length: {}'.format(len(data_df)))

    test_df = data_df[data_df['problem_id'].map(lambda x: x in test_problem_ids)]
    # 将数据集的索引重置为从0开始
    test_df = test_df.reset_index(drop=True)
    print('test data size: {}'.format(test_df.shape[0]))
    return None, None, test_df


def read_codeflaws_df() -> pd.DataFrame:
    import config
    if not os.path.exists(config.CODEFLAWS_BENCHMARK_df_target):
        df = read_and_pickle_codeflaws_data()
    else:
        df = pd.read_pickle(config.CODEFLAWS_BENCHMARK_df_target)
    return df


@disk_cache(basename='read_fake_common_deepfix_error_dataset_with_limit_length', directory=CACHE_DATA_PATH)
def read_fake_common_deepfix_error_dataset_with_limit_length(limit_length=500):
    data_df = read_fake_deepfix_common_error_records()

    tokenize_fn = tokenize_by_clex_fn()
    data_df = filter_length(data_df, limit_length, tokenize_fn)
    print('after filter code length: {}'.format(len(data_df)))

    valid_df = data_df.sample(frac=0.05)
    data_df = data_df.drop(valid_df.index)
    test_df = data_df.sample(frac=0.05)
    train_df = data_df.drop(test_df.index)

    return train_df, valid_df, test_df


if __name__ == '__main__':
    # df = read_codeflaws_df()
    # df = df.sample(100)
    # print(df.columns)
    # for i, row in df.head(1).iterrows():
    #     print(row['problem_id'])
    #     print('right_code_id', row['right_code_id'])
    #     print(row['right_code'])
    #     print('error_code_id', row['error_code_id'])
    #     print(row['error_code'])
    #     print(row['test_case'])
    #     print(row['heldout_test_case'])
    #     print(len(row['test_case']))
    #     print(len(row['heldout_test_case']))

    train_df, valid_df, test_df = read_codeforces_real_semantic_python_dataset()
    # print(len(train_df), len(valid_df), len(test_df))
    print(len(test_df))

    # tokenize_fn = create_python_tokenize_fn()
    # train_df['change_record'] = train_df['change_record'].map(json.loads)
    # train_df['before_length'] = train_df['change_record'].map(lambda x: len(tokenize_fn(x['original'])))
    # train_df['after_length'] = train_df['change_record'].map(lambda x: len(tokenize_fn(x['after'])))
    # print('max_before_length: ', max(train_df['before_length']))
    # print('max_after_length: ', max(train_df['after_length']))
    #
    # for i, row in train_df.iterrows():
    #     if row['before_length'] > 100 or row['after_length'] > 100:
    #         print(i, row)


    # before_length = train_df['before_length'].tolist()
    # after_length = train_df['after_length'].tolist()

    before_bins = []




