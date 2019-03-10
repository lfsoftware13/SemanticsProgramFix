import sqlite3
import pandas as pd
import json

from common.constants import langdict, verdict, CACHE_DATA_PATH, scrapyOJ_DB_PATH, COMMON_DEEPFIX_ERROR_RECORDS
from common.util import disk_cache
from config import FAKE_DEEPFIX_ERROR_DATA_DBPATH, FAKE_CODEFORCES_PYTHON_DATA_DBPATH


# 读取python数据，以DataFrame格式返回
def read_python_data_artificalCode():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_CODEFORCES_PYTHON_DATA_DBPATH), uri=True)
    df = pd.read_sql('select * from artificalCode', conn)
    conn.close()
    return df

def read_data(conn, table, condition=None):
    extra_filter = ''
    note = '"'
    if condition is not None:
        extra_filter += ' where '
        condition_str = ['{}{}{}'.format(con[0], con[1], con[2]) for con in condition]
        extra_filter += (' and '.join(condition_str))
    sql = 'select * from {} {}'.format(table, extra_filter)
    data_df = pd.read_sql(sql, conn)
    print('read data sql statment: {}. length:{}'.format(sql, len(data_df.index)))
    return data_df


def merge_and_deal_submit_table(problems_df, submit_df):
    submit_joined_df = submit_df.join(problems_df.set_index('problem_name'), on='problem_name')
    submit_joined_df['time'] = submit_joined_df['time'].str.replace('ms', '').astype('int')
    submit_joined_df['memory'] = submit_joined_df['memory'].str.replace('KB', '').astype('int')
    submit_joined_df['submit_time'] = pd.to_datetime(submit_joined_df['submit_time'])
    submit_joined_df['tags'] = submit_joined_df['tags'].str.split(':')
    submit_joined_df['code'] = submit_joined_df['code'].str.slice(1, -1)
    submit_joined_df['language'] = submit_joined_df['language'].replace(langdict)
    submit_joined_df['status'] = submit_joined_df['status'].replace(verdict)
    return submit_joined_df


def read_cpp_testcase_error_records_from_db(conn):
    df = pd.read_sql('select * from cpp_testcase_error_records', conn)
    return df


def read_problem_testcase_error_records(conn):
    df = pd.read_sql('select * from problem_testcase', conn)
    return df


def read_all_python_data(conn):
    problems_df = pd.read_sql('select problem_name, tags from {}'.format('problem'), conn)
    submit_df = pd.read_sql('select * from {} where language="Python 3"'.format('submit'), conn)
    submit_joined_df = merge_and_deal_submit_table(problems_df, submit_df)
    return submit_joined_df


@disk_cache(basename='read_all_python_records', directory=CACHE_DATA_PATH)
def read_all_python_records():
    conn = sqlite3.connect("file:{}?mode=ro".format(scrapyOJ_DB_PATH), uri=True)
    data_df = read_all_python_data(conn)
    return data_df


@disk_cache(basename='read_fake_common_deepfix_error_records', directory=CACHE_DATA_PATH)
def read_fake_common_deepfix_error_records():
    conn = sqlite3.connect('file:{}?mode=ro'.format(FAKE_DEEPFIX_ERROR_DATA_DBPATH), uri=True)
    data_df = read_data(conn, COMMON_DEEPFIX_ERROR_RECORDS)
    return data_df


def load_data(s):
    # print('start: ', s[1:-1])
    return json.loads(s[1:-1])


if __name__ == '__main__':
    # conn = sqlite3.connect('data/train_data.db')
    # df = read_cpp_testcase_error_records_from_db(conn)
    # df = df[df['testcase'].map(lambda x: x != '')]
    # df['testcase'] = df['testcase'].map(json.loads)

    df = read_python_data_artificalCode()
    print(len(df))
    print(df.columns)
    print(df.iloc[0])

    # prob_conn = sqlite3.connect(r'C:/Users/Lf/Desktop/problem_testcase.db')
    # df = read_problem_testcase_error_records(prob_conn)
    # df = df[df['testcase'].map(lambda x: x != '' and x != "'[]'")]
    # df['testcase'] = df['testcase'].map(load_data)
    # df = df[df['testcase'].map(lambda x: len(x) > 0)]
    # df['total_testcase'] = df['testcase'].map(len)
    # df['effect_testcase'] = df['testcase'].map(lambda x: len(list(filter(lambda y: '...' != y['input'][-3:] and
    #                                                                            '...' != y['output'][-3:] and
    #                                                                            '...' != y['answer'][-3:], x))))
    # print('total_testcase: {}, effect_testcase: {}'.format(sum(df['total_testcase']), sum(df['effect_testcase'])))
