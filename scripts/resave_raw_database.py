from common.constants import C_SUBMIT_TABLE
from config import c_db_path, scrapyOJ_path
from database.database_util import insert_items, create_table

import sqlite3
import pandas as pd


def add_params(params, need_quot=True):
    params_str = ''
    is_first = True
    if len(params.keys()) > 0:
        for k, v in params.items():
            if v is not None:
                if not is_first:
                    params_str += ' and '
                is_first = False
                if need_quot:
                    params_str += ' {}="{}" '.format(k, v)
                else:
                    params_str += ' {}={} '.format(k, v)
    return params_str


def resave_database_main(to_db_path, to_table_name, params_string: dict={}, params_number: dict={}):

    params_s = add_params(params_string, need_quot=True)
    params_n = add_params(params_number, need_quot=False)

    if params_s != '' and params_n != '':
        params = params_s + ' and ' + params_n
    else:
        params = params_s if params_s != '' else params_n
    parmas = ' where ' + params if params != '' else params

    conn = sqlite3.connect(scrapyOJ_path)
    print('start read sql')
    df = pd.read_sql('select * from {}{}'.format('submit', parmas), conn)
    print('total df length: {}'.format(len(df)))
    df = df[df['code'].map(lambda x: x != '')]
    print('no empty df length: {}'.format(len(df)))
    df_dict = df.to_dict(orient='list')
    del df

    print('finish filter')
    create_table(to_db_path, to_table_name)
    header_list = ['id', 'submit_url', 'submit_time', 'user_id', 'user_name', 'problem_id', 'problem_url', 'problem_name', 'problem_full_name', 'language', 'status', 'error_test_id', 'time', 'memory', 'code']
    total_list = [df_dict[key] for key in header_list]
    total_list = list(zip(*total_list))
    print('start save')
    insert_items(to_db_path, to_table_name, total_list)
    print('end save')


if __name__ == '__main__':
    resave_database_main(c_db_path, C_SUBMIT_TABLE, params_string={'language': 'GNU C11'})
