from common.util import show_process_map, CustomerDataSet
import pandas as pd

from vocabulary.word_vocabulary import Vocabulary


class CombineNodeIterateErrorDataSet(CustomerDataSet):
    def __init__(self,
                 data_df: pd.DataFrame,
                 vocabulary: Vocabulary,
                 set_type: str,
                 transformer_vocab_slk=None,
                 no_filter=False,
                 do_flatten=False,
                 MAX_LENGTH=500,
                 use_ast=False,
                 do_multi_step_sample=False,
                 id_to_program_dict=None,
                 no_id_to_program_dict=False):
        # super().__init__(data_df, vocabulary, set_type, transform, no_filter)
        self.set_type = set_type
        self.vocabulary = vocabulary
        self.transformer = transformer_vocab_slk
        self.is_flatten = do_flatten
        self.max_length = MAX_LENGTH
        self.use_ast = use_ast
        self.transform = False
        self.id_to_program_dict = id_to_program_dict
        if no_id_to_program_dict:
            self.id_to_program_dict = None
        # if self.set_type != 'valid' and self.set_type != 'test' and self.set_type != 'deepfix':
        #     self.do_sample = False
        # else:
        #     self.do_sample = True
        self.do_multi_step_sample = do_multi_step_sample
        if data_df is not None:
            if not no_filter:
                self.data_df = self.filter_df(data_df)
            else:
                self.data_df = data_df

            if id_to_program_dict is None:
                self.id_to_program_dict = {i: prog_id for i, prog_id in enumerate(sorted(data_df['id']))}
            else:
                self.id_to_program_dict = id_to_program_dict
            if no_id_to_program_dict:
                self.id_to_program_dict = None
            self.only_first = do_multi_step_sample
            self._samples = [FlattenRandomIterateRecords(row, is_flatten=do_flatten, only_first=do_multi_step_sample)
                             for i, row in self.data_df.iterrows()]
            # c = 0
            # for i, (index, row) in self.data_df.iterrows():
            #     print(i)
            #     print(row['id'])
            self.program_to_position_dict = {row['id']: i for i, (index, row) in enumerate(self.data_df.iterrows())}

            if self.transform:
                self._samples = show_process_map(self.transform, self._samples)
            # for s in self._samples:
            #     for k, v in s.items():
            #         print("{}:shape {}".format(k, np.array(v).shape))

    def filter_df(self, df):
        df = df[df['error_token_id_list'].map(lambda x: x is not None)]
        # print('CCodeErrorDataSet df before: {}'.format(len(df)))
        df = df[df['distance'].map(lambda x: x >= 0)]

        def iterate_check_max_len(x):
            if not self.is_flatten:
                for i in x:
                    if len(i) > self.max_length:
                        return False
                return True
            else:
                return len(x) < self.max_length
        df = df[df['error_token_id_list'].map(iterate_check_max_len)]
        end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[0])

        return df

    def _get_raw_sample(self, row):
        # sample = dict(row)
        row.select_random_i(only_first=self.only_first)
        sample = {}
        sample['id'] = row['id']
        sample['includes'] = row['includes']
        # if not self.is_flatten and self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list'][0]
        #     sample['input_seq_name'] = row['error_token_name_list'][0][1:-1]
        #     sample['input_length'] = len(sample['input_seq'])
        # elif not self.is_flatten and not self.do_multi_step_sample:
        #     sample['input_seq'] = row['error_token_id_list']
        #     sample['input_seq_name'] = [r[1:-1] for r in row['error_token_name_list']]
        #     sample['input_length'] = [len(ids) for ids in sample['input_seq']]
        # else:
        sample['input_seq'] = row['error_token_id_list']
        sample['input_seq_name'] = row['error_token_name_list'][1:-1]
        sample['input_length'] = len(sample['input_seq'])
        sample['copy_length'] = sample['input_length']
        sample['adj'] = 0

        inner_begin_id = self.vocabulary.word_to_id(self.vocabulary.begin_tokens[1])
        inner_end_id = self.vocabulary.word_to_id(self.vocabulary.end_tokens[1])
        if not self.do_multi_step_sample:
            sample['target'] = [inner_begin_id] + row['sample_ac_id_list'] + [inner_end_id]

            sample['is_copy_target'] = row['is_copy_list'] + [0]
            sample['copy_target'] = row['copy_pos_list'] + [-1]

            sample_mask = sorted(row['sample_mask_list'] + [inner_end_id])
            sample_mask_dict = {v: i for i, v in enumerate(sample_mask)}
            sample['compatible_tokens'] = [sample_mask for i in range(len(sample['is_copy_target']))]
            sample['compatible_tokens_length'] = [len(one) for one in sample['compatible_tokens']]

            sample['sample_target'] = row['sample_ac_id_list'] + [inner_end_id]
            sample['sample_target'] = [t if c == 0 else -1 for c, t in zip(sample['is_copy_target'], sample['sample_target'])]
            sample['sample_small_target'] = [sample_mask_dict[t] if c == 0 else -1 for c, t in zip(sample['is_copy_target'], sample['sample_target'])]
            sample['sample_outputs_length'] = len(sample['sample_target'])

            sample['full_output_target'] = row['target_ac_token_id_list'][1:-1]

            sample['final_output'] = row['ac_code_ids']
            sample['final_output_name'] = row['ac_code_name_with_labels'][1:-1]
            sample['p1_target'] = row['error_pos_list'][0]
            sample['p2_target'] = row['error_pos_list'][1]
            sample['error_pos_list'] = row['error_pos_list']

            sample['distance'] = row['distance']
            sample['includes'] = row['includes']
        else:
            pass

        return sample

    def add_samples(self, df):
        df = self.filter_df(df)
        self._samples += [row for i, row in df.iterrows()]

    def remain_samples(self, count=0, frac=1.0):
        if count != 0:
            self._samples = random.sample(self._samples, count)
        elif frac != 1:
            count = int(len(self._samples) * frac)
            self._samples = random.sample(self._samples, count)

    def combine_dataset(self, dataset):
        d = IterateErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                              transformer_vocab_slk=self.transformer)
        d._samples = self._samples + dataset._samples
        return d

    def remain_dataset(self, count=0, frac=1.0):
        d = IterateErrorDataSet(data_df=None, vocabulary=self.vocabulary, set_type=self.set_type,
                                transformer_vocab_slk=self.transformer)
        d._samples = self._samples
        d.remain_samples(count=count, frac=frac)
        return d

    def __getitem__(self, index):
        if self.id_to_program_dict is not None:
            prog_id = self.id_to_program_dict[index]
            real_position = self.program_to_position_dict[prog_id]
        else:
            real_position = index
        row = self._samples[real_position]
        return self._get_raw_sample(row)

    def __setitem__(self, key, value):
        if self.id_to_program_dict is not None:
            prog_id = self.id_to_program_dict[key]
            real_position = self.program_to_position_dict[prog_id]
        else:
            real_position = key
        self._samples[real_position] = value

    def __len__(self):
        return len(self._samples)
