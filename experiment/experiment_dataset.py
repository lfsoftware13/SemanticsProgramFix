import random

from common.util import show_process_map, CustomerDataSet
import pandas as pd

from experiment.experiment_util import load_fake_deepfix_dataset_iterate_error_data
from vocabulary.word_vocabulary import Vocabulary


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
