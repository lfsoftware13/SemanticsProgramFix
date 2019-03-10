import json
import sqlite3
import pandas as pd

from config import FAKE_CODEFORCES_PYTHON_DATA_DBPATH


# convert eval python string to json string
def main():
    conn = sqlite3.connect(FAKE_CODEFORCES_PYTHON_DATA_DBPATH)
    print(FAKE_CODEFORCES_PYTHON_DATA_DBPATH)
    df = pd.read_sql('select id, change_record from artificalCode', conn)
    print(len(df))
    print(df['change_record'].iloc[0])
    df['change_record'] = df['change_record'].map(eval)
    df['change_record'] = df['change_record'].map(json.dumps)
    o = df['change_record'].iloc[0]
    print(o)

    id_list = df['id'].tolist()
    change_record_list = df['change_record'].tolist()
    obj_list = [(c, i) for i, c in zip(id_list, change_record_list)]
    print(obj_list[0])

    sql = r'''UPDATE artificalCode SET change_record=? WHERE id=?'''
    conn.executemany(sql, obj_list)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    main()
