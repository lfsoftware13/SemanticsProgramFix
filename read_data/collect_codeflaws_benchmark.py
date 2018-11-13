import os
import re
from collections import defaultdict, OrderedDict
import tqdm

import pandas as pd

from common import util


def read_content(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


class TestCaseException(Exception):
    pass


def read_test_cases(root_path, pattern):
    """
    :param root_path: the root path of the test case directory
    :param pattern: the test case name prefix pattern
    :return: a list of pair (input_str, output_str)
    """
    res = defaultdict(lambda: [None, None])

    name_dict = {"input": 0, "output": 1}

    for p_path in util.scan_dir(root_path, pattern=pattern):
        _, name = os.path.split(p_path)
        m = pattern(name)
        input_output = m.group(1)
        test_case_id = m.group(2)
        res[test_case_id][name_dict[input_output]] = read_content(p_path)

    for _, (intput, output) in res.items():
        if intput is None or output is None:
            raise TestCaseException()

    res = list(res.values())

    return res


def collect(path, debug=False):
    """
    :param path: The root path of the benchmark
    :return: a list of tuple
        [problem_id, right_code_id, right_code, error_code_id, error_code, test_case, heldout_test_case]
        test_case is a list of pair (input_str, output_str)
        heldout_test_case is the same as the test_case
    """
    item_directory_name_pattern = re.compile(r"(\d+)-([ABCDEFGH])-bug-(\d+)-(\d+)$")
    heldout_test_case_pattern = re.compile(r"heldout-(input|output)-pos(\d+)")
    test_case_pattern = re.compile(r"(input|output)-pos(\d+)")
    res = []
    for p_path in tqdm.tqdm(util.scan_dir(path, dir_level=0)):
        if debug and len(res) > 10:
            break
        base_name, name = os.path.split(p_path)
        m = item_directory_name_pattern.match(name)
        if m:
            contest_id = m.group(1)
            problem_id = m.group(2)
            buggy_id = m.group(3)
            accepted_id = m.group(4)

            problem_str = "{}{}".format(contest_id, problem_id)

            buggy_code_name = os.path.join(p_path, "{}-{}-{}.c".format(contest_id, problem_id, buggy_id))
            buggy_content = read_content(buggy_code_name)
            accepted_code_name = os.path.join(p_path, "{}-{}-{}.c".format(contest_id, problem_id, accepted_id))
            accepted_content = read_content(accepted_code_name)

            try:
                test_case = read_test_cases(p_path, lambda x: test_case_pattern.match(x))
                heldout_test_case = read_test_cases(p_path, lambda x: heldout_test_case_pattern.match(x))
            except Exception as e:
                continue

            res.append((problem_str, buggy_id, buggy_content, accepted_id, accepted_content, test_case,
                        heldout_test_case))

    return res


def read_and_pickle_codeflaws_data():
    import config
    benchmark_path = config.CODEFLAWS_BENCHMARK
    benchmark_content = collect(benchmark_path, debug=False)
    name_list = ["problem_id", "error_code_id", "error_code", "right_code_id", "right_code", "test_case",
                 "heldout_test_case"]
    contents = [list(t) for t in zip(*benchmark_content)]

    df = pd.DataFrame(OrderedDict({
        k: v for k, v in zip(name_list, contents)
    }))
    # print(df)
    df.to_pickle(config.CODEFLAWS_BENCHMARK_df_target)
    return df


if __name__ == '__main__':
    read_and_pickle_codeflaws_data()
