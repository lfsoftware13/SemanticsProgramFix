import pandas as pd
import os

from common.constants import CACHE_DATA_PATH
from common.pycparser_util import tokenize_by_clex_fn
from common.util import disk_cache, filter_length
from read_data.collect_codeflaws_benchmark import read_and_pickle_codeflaws_data
from read_data.read_filter_data import read_fake_deepfix_common_error_records


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

    df = read_fake_common_deepfix_error_dataset_with_limit_length(500)[0]
    print(df.columns)


