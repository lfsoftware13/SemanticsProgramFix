# -*- coding: utf-8 -*-

"""
@Time : 2018/11/13 20:39
@Author : Panjks-
@Description :
"""

import json
import csv
import linecache
import os
import sys
import inspect
import time
import sqlite3
import logging
import threading
import timeout_decorator
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
                    filename='tracer.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s: %(message)s'
                    )

HANDLE_CODE_ANNOTATION = True
CREATE_TABLE_FLAG = True
state = {
    'speed': 'quick'
}

item_status = {
    'type': 'WRONG_ANSWER'
}

starting_filename = r"/home/panjks/SemanticsProgramFix/tests/temp_python_code.py"
starting_dir = os.path.dirname(starting_filename)

temp_testcase_input_filepath = r'/home/panjks/SemanticsProgramFix/tests/temp_testcase_input.txt'
temp_testcase_output_filepath = r'/home/panjks/SemanticsProgramFix/tests/temp_testcase_output.txt'
error_testcase_csv = r'/home/panjks/SemanticsProgramFix/tests/error.csv'

problem_db = sqlite3.connect("problem_testcase.db")
python_code_db = sqlite3.connect("/home/lf/new_disk/panjks/python_data.db")
# python_code_db = sqlite3.connect("python_data.db")

os.chdir(starting_dir)
sys.path.insert(0, starting_dir)

current_filename = None
current_line = None
current_locals = {}
failed = False

currrent_tracer = []

create_tracer_table_sql = r"""
CREATE TABLE IF NOT EXISTS tracer (
    id              TEXT PRIMARY KEY,
    submit_id       TEXT,
    submit_time     TEXT,
    submit_url      TEXT,
    user_id         TEXT,
    problem_id      TEXT,
    testcase_id     TEXT,
    testcase_input  TEXT,
    testcase_output TEXT,
    testcase_answer TEXT,
    testcase_status TEXT,
    tracer          TEXT
)"""


def debounce(wait):
    def decorator(fn):
        class context:
            last_call = None

        def debounced(*args, **kwargs):
            def call_it():
                args, kwargs = context.last_call
                fn(*args, **kwargs)
                context.last_call = None

            # if context.last_call is None:
            #     debounced.t = threading.Timer(wait, call_it)
            #     debounced.t.start()
            # context.last_call = (args, kwargs)

            if context.last_call is None:
                context.last_call = (args, kwargs)
                call_it()
            else:
                context.last_call = (args, kwargs)

        return debounced

    return decorator


def log(msg, target_list):
    target_list.append(msg)


@debounce(0.1)
def log_frame(frame):
    log(generate_call_event(frame), current_tracer)


def should_ignore_variable(name, code_str):
    # return 0
    return name.startswith('__') and name.endswith('__') and name not in code_str


def truncate_list(l):
    # if len(l) > 3:
    #     ret = ', '.join(map(process_variable, l[:2]))
    #     ret += ", ..., "
    #     ret += process_variable(l[-1])
    #     return ret
    # else:
    return ', '.join(map(process_variable, l))


def format_function(f):
    args = inspect.getargspec(f).args
    return "function(%s)" % truncate_list(args)


def format_list(l):
    return "[%s]" % truncate_list(l)


def process_variable(var):
    type_name = type(var).__name__
    if type_name == 'list':
        return format_list(var)
    elif type_name == 'module':
        return "<module '%s'>" % var.__name__
    else:
        return str(var)


def get_module_name(full_path):
    global starting_filename
    return os.path.relpath(
        os.path.abspath(full_path),
        os.path.dirname(os.path.abspath(starting_filename))
    )


def generate_call_event(frame):
    source_code = "".join(linecache.getlines(frame.f_code.co_filename))
    frame_locals = {k:
                        {'value': process_variable(v), 'type': type(v).__name__}
                    for k, v in frame.f_locals.items() if not should_ignore_variable(k, source_code)
                    }
    frame_globals = {k:
                         {'value': process_variable(v), 'type': type(v).__name__}
                     for k, v in frame.f_globals.items() if not should_ignore_variable(k, source_code)
                     }
    obj = {
        'type': 'call',
        'frame_locals': frame_locals,
        'frame_globals': frame_globals,
        'lineno': frame.f_lineno,
        # 'source': ''.join(linecache.getlines(frame.f_code.co_filename))
    }
    return obj


def generate_exception_event(e):
    return {
        'type': 'exception',
        'exception_type': type(e).__name__,
        'exception_message': str(e),
        'filename': current_filename,
        'lineno': current_line,
        'time': time.time()
    }


def process_msg(msg):
    global state
    if type(msg) == bytes:
        msg = msg.decode('utf8')
    msg = json.loads(msg)
    if msg['type'] == 'change_speed':
        print('changed speed')
        state['speed'] = msg['speed']


def local_trace(frame, why, arg):
    global current_line
    global current_filename

    if failed:
        return

    if why == 'exception':
        exc_type = arg[0].__name__
        exc_msg = arg[1]
        return

    current_filename = frame.f_code.co_filename
    current_line = frame.f_lineno

    if not current_filename.startswith(starting_dir):
        return

    if 'livepython' in current_filename:
        return

    if 'site-packages' in current_filename:
        return

    if 'lib/python' in current_filename:
        return

    log_frame(frame)

    if state['speed'] == 'slow':
        time.sleep(0.1)
    elif state['speed'] == 'quick':
        pass

    return local_trace


def global_trace(frame, why, arg):
    return local_trace


# def get_python_code_data():
#     python_code_data = python_code_db.execute("SELECT * FROM submit")
#     data_list = []
#     for row in python_code_data:
#         temp = dict()
#         temp["submit_time"] = row[2]
#         temp["submit_url"] = row[1]
#         temp["user_id"] = row[3]
#         temp["problem_id"] = row[5]
#         temp["code"] = row[14]
#         temp["status"] = row[10]
#         temp["submit_id"] = row[0]
#         data_list.append(temp)
#     return data_list


def get_python_code_data(id_list: list = None):
    data_list = []
    append_str = ''
    if id_list is not None and len(id_list) > 0:
        id_str_list = [str(i) for i in id_list]
        id_str = ', '.join(id_str_list)
        append_str = " where id in ({})".format(id_str)
    python_code_data = python_code_db.execute("SELECT * FROM submit " + append_str)
    for row in python_code_data:
        temp = dict()
        temp["submit_time"] = row[2]
        temp["submit_url"] = row[1]
        temp["user_id"] = row[3]
        temp["problem_id"] = row[5]
        temp["code"] = row[14]
        temp["status"] = row[10]
        temp["submit_id"] = row[0]
        data_list.append(temp)
    return data_list


def get_python_code_data_by_file(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            temp = json.loads(line.strip())
            data_list.append(temp)
    return data_list


@timeout_decorator.timeout(2, use_signals=False)
def get_tracer():
    sys.settrace(None)
    threading.settrace(None)
    run_flag = True
    global current_tracer
    current_tracer = []
    stdout_backup = sys.stdout
    log_file = open(temp_testcase_output_filepath, "w")
    sys.stdout = log_file
    with open(starting_filename, 'rb') as fp:
        code = compile(fp.read(), starting_filename, 'exec')
    namespace = {
        '__file__': starting_filename,
        '__name__': '__main__',
    }

    log({'type': 'start'}, current_tracer)

    sys.settrace(global_trace)
    threading.settrace(global_trace)

    try:
        with open(temp_testcase_input_filepath, 'r') as sys.stdin:
            exec(code, namespace)
            log({'type': 'finish'}, current_tracer)
    except Exception as err:
        run_flag = False
        logging.exception(str(err))
        # log(json.dumps(generate_exception_event(err)), current_tracer)
    except SystemExit:
        log({'type': 'finish'}, current_tracer)
    finally:
        sys.settrace(None)
        threading.settrace(None)
    log_file.close()
    sys.stdout = stdout_backup
    return current_tracer, run_flag


if __name__ == '__main__':
    csv_file = open(error_testcase_csv, 'w', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    if CREATE_TABLE_FLAG:
        python_code_db.execute(create_tracer_table_sql)
        logging.info("Tracer tables created successfully.")
        python_code_db.commit()
    # special_id = ['27758742']
    special_id = None
    python_data_dict = get_python_code_data(special_id)
    print(len(python_data_dict))
    with tqdm(total=len(python_data_dict)) as pbar:
        for item in python_data_dict:
            pbar.update(1)
            if "open" in item["code"] or "listdir" in item['code'] or 'urllib' in item['code'] or \
                    'request' in item['code']:
                logging.error("ITEM No.{} use special api.".format(item["submit_id"], item_status["type"]))
                continue
            if item['status'] != item_status["type"]:
                logging.error("ITEM No.{} status is not '{}'.".format(item["submit_id"], item_status["type"]))
                continue
            # if item['status'] != "OK":
            #     logging.error("ITEM No.{} status is not 'WRONG_ANSWER'.".format(item_id))
            #     continue
            with open(starting_filename, 'w', encoding='utf-8') as f:
                f.write(item["code"][1:-1])

            testcases = problem_db.execute(
                "SELECT testcase FROM problem_testcase WHERE problem_id = {}".format(item["problem_id"]))
            try:
                testcases = testcases.fetchone()[0][1:-1]
                testcases = json.loads(testcases)
            except TypeError:
                logging.error("ITEM No.{} failled to get testcases.".format(item["submit_id"]))
                continue

            testcase_id = 0
            for testcase in testcases:
                testcase_id += 1
                tracer_id = str(item['submit_id']) + "_" + str(testcase_id)

                tracer_existed = python_code_db.execute(
                    "SELECT * FROM tracer WHERE id = '{}'".format(tracer_id))
                try:
                    db_existed_flag = tracer_existed.fetchone()[0]
                    logging.error("ITEM No.{} status has existed in table 'tracer'.".format(tracer_id))
                    continue
                except TypeError:
                    pass
                if testcase["input"].endswith("...") or testcase["output"].endswith("..."):
                    logging.error(
                        "ITEM No.{} fail to get No.{} testcase because '...'.".format(item["submit_id"], testcase_id))
                    continue
                item["testcase_input"] = testcase["input"]
                item["testcase_answer"] = testcase["answer"]
                if len(testcase["input"]) >= 100:
                    logging.error(
                        "ITEM No.{} failed to get No.{} testcase tracer because testcase is too long.".format(
                            item["submit_id"],
                            testcase_id))
                    error_msg = [item["submit_id"], testcase_id, "testcase is too long."]
                    csv_writer.writerow(error_msg)
                    continue

                with open(temp_testcase_input_filepath, 'w', encoding='utf-8') as f:
                    f.write(testcase["input"])
                try:
                    tracer_result, code_run_flag = get_tracer()
                except Exception:
                    logging.error(
                        "ITEM No.{} failed to get No.{} testcase tracer because of timeout.".format(item["submit_id"],
                                                                                                    testcase_id))
                    error_msg = [item["submit_id"], testcase_id, "timeout."]
                    csv_writer.writerow(error_msg)
                    continue
                if not code_run_flag:
                    logging.error(
                        "ITEM No.{} failed to get No.{} testcase tracer because of exception.".format(item["submit_id"],
                                                                                                      testcase_id))
                    error_msg = [item["submit_id"], testcase_id, "code run error."]
                    csv_writer.writerow(error_msg)
                    continue
                item["tracer"] = json.dumps(tracer_result)
                with open(temp_testcase_output_filepath, 'r', encoding='utf-8') as f:
                    item["testcase_output"] = f.read().strip()
                    if item["testcase_output"] == testcase["answer"].strip():
                        item["testcase_status"] = "right"
                    else:
                        item["testcase_status"] = "wrong"
                try:
                    sql = '''INSERT INTO tracer (id,submit_id,submit_time,submit_url,user_id,problem_id,testcase_id,testcase_input,testcase_output,testcase_answer,testcase_status,tracer) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
                    python_code_db.execute(sql,
                                           (tracer_id, item["submit_id"], item["submit_time"], item["submit_url"],
                                            item["user_id"],
                                            item["problem_id"],
                                            testcase_id,
                                            item["testcase_input"],
                                            item["testcase_output"], item["testcase_answer"], item["testcase_status"],
                                            item["tracer"]))
                    logging.info(
                        "ITEM No.{} success to get No.{} testcase tracer. tracer_id = {}".format(item["submit_id"],
                                                                                                 testcase_id,
                                                                                                 tracer_id))

                except Exception:
                    logging.error(
                        "ITEM No.{} failed to get No.{} testcase tracer because of tracer too long.".format(
                            item["submit_id"],
                            testcase_id))
                    error_msg = [item["submit_id"], testcase_id, "tracer too long."]
                    csv_writer.writerow(error_msg)
                    continue
                python_code_db.commit()
    python_code_db.close()
    problem_db.close()
    csv_file.close()
