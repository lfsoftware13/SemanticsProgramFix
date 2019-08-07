import random
from read_data.read_experiment_data import python_df_to_dataset
from common.util import show_process_map, CustomerDataSet
import pandas as pd
from tokenize import tokenize
from io import BytesIO
from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data, \
    load_fake_semantics_deepfix_dataset, load_fake_semantic_python_dataframes, \
    load_codeforces_real_semantic_python_dataframes
from vocabulary.word_vocabulary import Vocabulary


# python提取需要的数据字段形成输入数据集
def python_get_dataset():
    df_tuple = python_df_to_dataset()
    dataset = [[], [], []]
    for i in range(3):
        df = df_tuple[i]
        for index, row in df.iterrows():
            item = dict()
            token_list = []
            tokens = tokenize(BytesIO(row['artificial_code'].encode('utf-8')).readline)
            for token in tokens:
                token_list.append(token[1])
            item['artificial_code_tokens'] = token_list

            change_record = eval(row['change_record'])
            token_list = []
            tokens = tokenize(BytesIO(change_record['after'].encode('utf-8')).readline)
            for token in tokens:
                if token[1] != 'utf-8':
                    token_list.append(token[1])
            item['change_line_tokens'] = token_list
            
            item['error_type'] = change_record['errorType']
            dataset[i].append(item)
    train_dataset, valid_dataset, test_dataset = dataset[0], dataset[1], dataset[2]
    print('最终数据集的长度', len(train_dataset), len(valid_dataset), len(test_dataset))
    #print(train_dataset[0]['change_line_tokens'], train_dataset[0]['error_type'])
    return train_dataset, valid_dataset, test_dataset


class LineSequenceDataset(CustomerDataSet):
    def __init__(self, data_df: pd.DataFrame, vocabulary: Vocabulary, name: str, do_sample: bool, MAX_LENGTH=500,
                 use_ast=False):
        self.vocabulary = vocabulary
        self.name = name
        self.do_sample = do_sample
        self.max_length = MAX_LENGTH
        self.use_ast = use_ast

        self.data_df = self.filter_df(data_df)
        self._samples = [row for _, row in data_df.iterrows()]

    def filter_df(self, df):
        df = df[df['error_token_ids'].map(lambda x: len(x) < self.max_length)]
        return df

    def _get_raw_sample(self, row):
        sample = {}
        sample['id'] = row['id']
        sample['code'] = row['artificial_code']
        sample['error_token_ids'] = row['error_token_ids']
        sample['error_token_length'] = len(sample['error_token_ids'])
        sample['error_line_token_length'] = row['error_line_token_length']
        sample['error_line_length'] = len(sample['error_line_token_length'])

        if not self.do_sample:
            sample['error_line_ids'] = row['change_after_tokens_ids']
            sample['target_line_ids'] = row['change_original_tokens_ids']
            sample['target_line_length'] = len(sample['target_line_ids'])
            sample['error_line'] = row['error_line']

        if self.use_ast:
            from common.python_parse_util import load_python_parse_graph
            st_list, input_sequence, adj = load_python_parse_graph(row['artificial_code'], adjacent_type='tuple')
            sample['error_token_ids'] = self.vocabulary.parse_text([input_sequence], use_position_label=False)[0]
            sample['error_token_length'] = len(sample['error_token_ids'])
            sample['adj'] = adj
        return sample

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __setitem__(self, key, value):
        self._samples[key] = value

    def __len__(self):
        return len(self._samples)


class SequenceCodeDataset(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 name:str,
                 do_sample: bool,
                 no_filter=False,
                 MAX_LENGTH=500,):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.do_sample = do_sample
        self.name = name
        self.vocabulary = vocabulary
        self.max_length = MAX_LENGTH
        self.transform = False
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df

            self._samples = [row for i, row in self.data_df.iterrows()]

            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)

    def filter_df(self, df):
        df = df[df['error_token_id_list'].map(lambda x: x is not None)]
        # print('SequenceCodeDataset df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]

        def check_max_len(x):
            return len(x) < self.max_length
        df = df[df['error_token_id_list'].map(check_max_len)]

        return df

    def _get_raw_sample(self, row):
        # sample = dict(row)
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        sample['distance'] = row['distance']

        sample['input_seq'] = row['error_token_id_list']
        sample['input_seq_name'] = row['error_token_name_list'][1:-1]
        sample['input_length'] = len(sample['input_seq'])

        sample['target_seq'] = row['target_token_id_list']
        sample['target_seq_name'] = row['target_token_name_list']
        sample['target_length'] = len(sample['target_seq'])

        return sample

    def __getitem__(self, index):
        return self._get_raw_sample(self._samples[index])

    def __setitem__(self, key, value):
        self._samples[key] = value

    def __len__(self):
        return len(self._samples)


def load_deepfix_sequence_dataset(is_debug, vocabulary, only_sample=False):

    data_dict = load_fake_deepfix_dataset_iterate_error_data(is_debug)

    datasets = [SequenceCodeDataset(pd.DataFrame(dd), vocabulary, name, do_sample=only_sample)
                for dd, name in zip(data_dict, ["train", "all_valid", "all_test"])]
    for d, n in zip(datasets, ["train", "val", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)
        # info(info_output)

    train_dataset, valid_dataset, test_dataset = datasets
    return train_dataset, valid_dataset, test_dataset


def load_deepfix_semantics_dataset(is_debug, vocabulary, only_sample=False):
    dfs = load_fake_semantics_deepfix_dataset(is_debug)

    datasets = [LineSequenceDataset(df, vocabulary, name, do_sample=only_sample)
                for df, name in zip(dfs, ['train', 'valid', 'test'])]
    for d, n in zip(datasets, ["train", "valid", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)

    return datasets


def load_fake_python_semantics_dataset(is_debug, vocabulary, max_sample_length, use_ast=False, only_sample=False):
    dfs = load_fake_semantic_python_dataframes(is_debug, max_sample_length)

    datasets = [LineSequenceDataset(df, vocabulary, name, do_sample=only_sample, use_ast=use_ast)
                for df, name in zip(dfs, ['train', 'valid', 'test'])]
    for d, n in zip(datasets, ["train", "valid", "test"]):
        info_output = "There are {} parsed data in the {} dataset".format(len(d), n)
        print(info_output)

    return datasets


def load_codeforces_real_python_semantics_dataset(is_debug, vocabulary, max_sample_length, use_ast=False, only_sample=False):
    _, _, test_df = load_codeforces_real_semantic_python_dataframes(is_debug, max_sample_length)

    name = 'test'
    test_dataset = LineSequenceDataset(test_df, vocabulary, name, do_sample=only_sample, use_ast=use_ast)
    info_output = "There are {} parsed data in the {} dataset".format(len(test_dataset), name)
    print(info_output)

    return None, None, test_dataset
