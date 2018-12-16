# -*- coding: utf-8 -*-

from tokenize import tokenize, untokenize, NUMBER, STRING, NAME, OP, N_TOKENS
from io import BytesIO
import sqlite3
from tqdm import tqdm


def remove_code_comments(code):
    result = []
    g = tokenize(BytesIO(code.encode('utf-8')).readline)
    for toknum, tokval, _, _, _ in g:
        if toknum != 57:
            result.append(
                (toknum, tokval)
            )
    # result = result[1:]
    return untokenize(result).decode('utf-8')


if __name__ == '__main__':
    python_code_db = sqlite3.connect("python_data.db")
    python_code_raw = python_code_db.execute("SELECT id,code FROM submit WHERE status = 'WRONG_ANSWER'")
    python_code_raw = python_code_raw.fetchall()
    update_sql = "UPDATE submit SET code = ? where id = ? "

    with tqdm(total=len(python_code_raw)) as pbar:
        for submit_id, raw_code in python_code_raw:
            raw_code = raw_code[1:-1]
            code_handled = remove_code_comments(raw_code)
            code_handled = "'" + code_handled + "'"
            python_code_db.execute(update_sql, (code_handled, submit_id))
            pbar.update(1)
    python_code_db.commit()
    python_code_db.close()
