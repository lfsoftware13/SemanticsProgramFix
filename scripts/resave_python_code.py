from common.constants import PYTHON_SUBMIT_TABLE
from common.util import init_code
from config import python_db_path, scrapyOJ_path
from database.database_util import insert_items, create_table
from read_data.read_data_from_db import read_all_python_records

import sqlite3
import pandas as pd


def resave_python_code_main():
    conn = sqlite3.connect(scrapyOJ_path)
    print('start read sql')
    df = pd.read_sql('select * from {} where language="Python 3"'.format('submit'), conn)
    print('total df length: {}'.format(len(df)))
    df = df[df['code'].map(lambda x: x != '')]
    print('no empty df length: {}'.format(len(df)))
    df_dict = df.to_dict(orient='list')
    del df

    print('finish filter')
    create_table(python_db_path, PYTHON_SUBMIT_TABLE)
    header_list = ['id', 'submit_url', 'submit_time', 'user_id', 'user_name', 'problem_id', 'problem_url', 'problem_name', 'problem_full_name', 'language', 'status', 'error_test_id', 'time', 'memory', 'code']
    total_list = [df_dict[key] for key in header_list]
    total_list = list(zip(*total_list))
    print('start save')
    insert_items(python_db_path, PYTHON_SUBMIT_TABLE, total_list)
    print('end save')


if __name__ == '__main__':
    resave_python_code_main()
