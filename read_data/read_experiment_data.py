import pandas as pd
import os
from tokenize import tokenize
from io import BytesIO
from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache, filter_length
from read_data.collect_codeflaws_benchmark import read_and_pickle_codeflaws_data
from read_data.read_filter_data import read_fake_deepfix_common_error_records, filter_distinct_artificalCode, python_filter


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

    df = python_df_to_dataset(500)[0]
    print(len(df))


