from common.analyse_include_util import extract_include, replace_include_with_blank
from common.constants import CACHE_DATA_PATH
from common.util import disk_cache, group_df_to_grouped_list, init_code
from read_data.read_data_from_db import read_fake_common_deepfix_error_records


def filter_distinct_table_key(data_df, key, max_num=None):
    if max_num is None:
        max_num = float('inf')
    group_list = group_df_to_grouped_list(data_df, key)
    print('group_list', len(group_list))
    num = min(max_num, len(group_list[0]))
    group_res = group_list[0].sample(num).copy(deep=True)
    i = 0
    for group in group_list[1:]:
        print('filter_distinct_table_key: {} in {}'.format(i, len(group_list)))
        i += 1
        num = min(max_num, len(group))
        group_res = group_res.append(group.sample(num), ignore_index=True)
    return group_res


def filter_distinct_problem_user_id(data_df):
    data_df = filter_distinct_table_key(data_df, 'problem_user_id', max_num=1)
    return data_df


def filter_distinct_problem(data_df, max_num=None):
    data_df = filter_distinct_table_key(data_df, 'problem_id', max_num=max_num)
    return data_df


def filter_distinct_user(data_df, max_num=None):
    data_df = filter_distinct_table_key(data_df, 'user_id', max_num=max_num)
    return data_df


def filter_distinct_test_c_data(data_df):
    data_df = filter_distinct_problem_user_id(data_df)
    data_df = filter_distinct_problem(data_df, 10)
    data_df = filter_distinct_user(data_df, 10)
    return data_df


@disk_cache(basename='read_fake_deepfix_common_error_records', directory=CACHE_DATA_PATH)
def read_fake_deepfix_common_error_records():
    data_df = read_fake_common_deepfix_error_records()
    print('origin data size: ', len(data_df))
    data_df = data_df[data_df['distance'].map(lambda x: 0 < x < 10)]
    print('after filter distance length between 0 and 10: ', len(data_df))
    data_df['includes'] = data_df['similar_code'].map(extract_include)
    data_df['similar_code'] = data_df['similar_code'].map(init_code)
    data_df['similar_code_with_includes'] = data_df['similar_code']
    data_df['similar_code'] = data_df['similar_code'].map(replace_include_with_blank).map(lambda x: x.replace('\r', ''))

    data_df['code'] = data_df['code'].map(init_code)
    data_df['code_with_includes'] = data_df['code']
    data_df['code'] = data_df['code'].map(replace_include_with_blank).map(lambda x: x.replace('\r', ''))

    data_df = data_df[data_df['similar_code'].map(lambda x: x != '')]
    return data_df