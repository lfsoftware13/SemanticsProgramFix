import json
from abc import abstractmethod, ABCMeta

import torch

from common.constants import LINE_TOKEN_SAMPLE_RECORDS
from common.problem_util import to_cuda
from common.torch_util import expand_tensor_sequence_to_same
from common.util import PaddedList

import torch.nn.functional as F

from database.database_util import create_table, run_sql_statment


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def clear_result(self):
        pass

    @abstractmethod
    def add_result(self, output_ids, model_output, model_target, model_input, batch_data):
        """

        :param output_ids:
        :param model_output: [batch, ..., vocab_size]
        :param model_target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        pass

    @abstractmethod
    def get_result(self):
        pass


class SequenceF1Score(Evaluator):
    """
    F1 score evaluator using in paper (A Convolutional Attention Network for Extreme Summarization of Source Code)
    """

    def __init__(self, vocab, rank=1):
        """
        Precision = TP/TP+FP
        Recall = TP/TP+FN
        F1 Score = 2*(Recall * Precision) / (Recall + Precision)
        :param rank: default 1
        :param ignore_token:
        :param gpu_index:
        """
        self.vocab = vocab
        self.rank = rank
        self.tp_count = 0
        # predict_y = TP + FP
        # actual_y = TP + FN
        self.predict_y = 0
        self.actual_y = 0

    def add_result(self, log_probs, model_target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: must be 3 dim. [batch, sequence, vocab_size]
        :param model_target:
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if isinstance(model_target, torch.Tensor):
            model_target = model_target.cpu()
            model_target = model_target.view(model_target.shape[0], -1)
            model_target = model_target.tolist()

        log_probs = log_probs.cpu()
        log_probs = log_probs.view(log_probs.shape[0], -1, log_probs.shape[-1])
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        end_id = self.vocab.word_to_id(self.vocab.end_tokens[0])
        unk_id = self.vocab.word_to_id(self.vocab.unk)
        batch_tp_count = 0
        batch_predict_y = 0
        batch_actual_y = 0
        for one_predict, one_target in zip(top_ids, model_target):
            one_predict, _ = self.filter_token_ids(one_predict, end_id, unk_id)
            one_target, _ = self.filter_token_ids(one_target, end_id, unk_id)
            one_tp = set(one_predict) & set(one_target)
            batch_tp_count += len(one_tp)
            batch_predict_y += len(one_predict)
            batch_actual_y += len(one_target)
        self.tp_count += batch_tp_count
        self.predict_y += batch_predict_y
        self.actual_y += batch_actual_y
        precision = float(batch_tp_count ) / float(batch_predict_y)
        recall = float(batch_tp_count) / float(batch_actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def filter_token_ids(self, token_ids, end, unk):
        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))
        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def clear_result(self):
        self.tp_count = 0
        self.predict_y = 0
        self.actual_y = 0

    def get_result(self):
        precision = float(self.tp_count) / float(self.predict_y)
        recall = float(self.tp_count) / float(self.actual_y)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.
        return f1

    def __str__(self):
        return ' SequenceF1Score top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceOutputIDToWord(Evaluator):
    def __init__(self, vocab, ignore_token=None, file_path=None):
        self.vocab = vocab
        self.ignore_token = ignore_token
        self.file_path = file_path
        if file_path is not None:
            with open(file_path, 'w') as f:
                pass

    def add_result(self, log_probs, model_target, ignore_token=None, gpu_index=None, batch_data=None):
        """

        :param log_probs: [batch, seq, vocab_size]
        :param model_target: [batch, seq]
        :param ignore_token:
        :param gpu_index:
        :param batch_data:
        :return:
        """
        if self.file_path is None:
            return
        if isinstance(model_target, torch.Tensor):
            model_target = model_target.cpu()
            model_target = model_target.tolist()

        log_probs = log_probs.cpu()
        _, top_ids = torch.max(log_probs, dim=-1)
        top_ids = top_ids.tolist()

        input_text = batch_data["text"]

        for one_input, one_top_id, one_target in zip(input_text, top_ids, model_target):
            predict_token = self.convert_one_token_ids_to_code(one_top_id, self.vocab.id_to_word)
            target_token = self.convert_one_token_ids_to_code(one_target, self.vocab.id_to_word)
            self.save_to_file(one_input, predict_token, target_token)

    def save_to_file(self, input_token=None, predict_token=None, target_token=None):
        if self.file_path is not None:
            with open(self.file_path, 'a') as f:
                f.write('---------------------------------------- one record ----------------------------------------\n')
                if input_token is not None:
                    f.write('input: \n')
                    f.write(str(input_token) + '\n')
                if predict_token is not None:
                    f.write('predict: \n')
                    f.write(predict_token + '\n')
                if target_token is not None:
                    f.write('target: \n')
                    f.write(target_token + '\n')

    def filter_token_ids(self, token_ids, start, end, unk):

        def filter_special_token(token_ids, val):
            return list(filter(lambda x: x != val, token_ids))

        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position+1]
        except ValueError as e:
            end_position = None
        # token_ids = filter_special_token(token_ids, start)
        # token_ids = filter_special_token(token_ids, end)
        token_ids = filter_special_token(token_ids, unk)
        return token_ids, end_position

    def convert_one_token_ids_to_code(self, token_ids, id_to_word_fn):
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        # token_ids, _ = self.filter_token_ids(token_ids, start, end, unk)
        tokens = [id_to_word_fn(tok) for tok in token_ids]
        code = ', '.join(tokens)
        return code

    def clear_result(self):
        pass

    def get_result(self):
        pass

    def __str__(self):
        return ''

    def __repr__(self):
        return self.__str__()


class SequenceBinaryClassExactMatch(Evaluator):

    def __init__(self, rank=1, ignore_token=None, gpu_index=None):
        self.rank = rank
        self.batch_count = 0
        self.match_count = 0
        self.ignore_token = ignore_token
        self.gpu_index = gpu_index

    def clear_result(self):
        self.batch_count = 0
        self.match_count = 0

    def add_result(self, output_ids, model_output, model_target, model_input, batch_data=None):
        """

        :param model_output: [batch, ...]
        :param model_target: [batch, ...], LongTensor, padded with target token. target.shape == log_probs.shape[:-1]
        :param ignore_token: optional, you can choose special ignore token and gpu index for one batch.
                            or use global value when ignore token and gpu_index is None
        :param gpu_index:
        :return:
        """
        top1_id = torch.gt(model_output, 0.5).long()
        # top1_id = torch.squeeze(top1_id, dim=-1)

        not_equal_result = torch.ne(top1_id, model_target)

        if self.ignore_token is not None:
            target_mask = torch.ne(model_target, self.ignore_token)
            not_equal_result = not_equal_result & target_mask
        batch_error_count = not_equal_result
        for i in range(len(not_equal_result.shape)-1):
            batch_error_count = torch.sum(batch_error_count, dim=-1)

        # [batch]
        batch_result = torch.eq(batch_error_count, 0)
        batch_match_count = torch.sum(batch_result).data.item()

        batch_size = model_output.shape[0]
        self.batch_count += batch_size
        self.match_count += batch_match_count
        return batch_match_count / batch_size

    def get_result(self):
        if self.batch_count == 0:
            return 0
        return self.match_count / self.batch_count

    def __str__(self):
        return ' SequenceBinaryClassExactMatch top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class TokenAccuracy(Evaluator):
    def __init__(self, ignore_token=None):
        self.total_count = 0.0
        self.match_count = 0.0
        self.ignore_token = ignore_token

    def clear_result(self):
        self.total_count = 0.0
        self.match_count = 0.0

    def add_result(self, output, model_output, model_target, model_input, batch_data=None):
        output_mask = torch.ne(model_target, self.ignore_token)
        count = torch.sum(output_mask).item()
        match = torch.sum(torch.eq(output, model_target) & output_mask).float().item()
        self.total_count += count
        self.match_count += match
        return match/count

    def get_result(self):
        if self.total_count == 0:
            return 0
        return self.match_count / self.total_count

    def __str__(self):
        return ' TokenAccuracy top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class SequenceCorrect(Evaluator):
    def __init__(self, ignore_token=None):
        self.total_batch = 0.0
        self.match_batch = 0.0
        self.ignore_token = ignore_token

    def clear_result(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def add_result(self, output, model_output, model_target, model_input=None, batch_data=None):
        output_mask = torch.ne(model_target, self.ignore_token)
        not_equal_batch = torch.sum(torch.ne(output, model_target) & output_mask, dim=-1).float()
        match = torch.sum(torch.eq(not_equal_batch, 0)).float()
        batch_size = output.shape[0]
        self.total_batch += batch_size
        self.match_batch += match.item()
        return match/batch_size

    def get_result(self):
        if self.total_batch == 0:
            return 0
        return self.match_batch / self.total_batch

    def __str__(self):
        return ' SequenceCorrect top 1: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class CompileResultEvaluate(Evaluator):
    """
    statistics compile result. It is a special evaluator. Do not use it in evaluator_list directly.
    """
    def __init__(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def clear_result(self):
        self.total_batch = 0.0
        self.match_batch = 0.0

    def add_result(self, result_list):
        match = sum(result_list)
        batch_size = len(result_list)
        self.match_batch += match
        self.total_batch += batch_size
        return 'step compile result: {}'.format(match/batch_size)

    def get_result(self):
        if self.total_batch == 0:
            return 0
        return self.match_batch / self.total_batch

    def __str__(self):
        return ' CompileResultEvaluate: ' + str(self.get_result())

    def __repr__(self):
        return self.__str__()


class LineTokenEvaluator(Evaluator):
    def __init__(self, ignore_token=None):
        self.ignore_token = ignore_token
        self.position_accuracy = TokenAccuracy(ignore_token=ignore_token)
        self.token_accuracy = TokenAccuracy(ignore_token=ignore_token)

    def add_result(self, output_ids, model_output, model_target, model_input, batch_data):
        output_position = torch.squeeze(torch.topk(F.log_softmax(model_output[0], dim=-1), k=1, dim=-1)[1], dim=-1)
        output_token_ids = torch.squeeze(torch.topk(F.log_softmax(model_output[1], dim=-1), k=1, dim=-1)[1], dim=-1)
        position_acc = self.position_accuracy.add_result(output_position, model_output[0], model_target[0], model_input, batch_data)
        output_token_acc = self.token_accuracy.add_result(output_token_ids, model_output[1], model_target[1], model_input, batch_data)
        return 'LineTokenEvaluator position_accuracy: {}, token_accuracy: {}'.format(position_acc, output_token_acc)

    def clear_result(self):
        self.position_accuracy.clear_result()
        self.token_accuracy.clear_result()

    def get_result(self):
        return self.position_accuracy.get_result(), self.token_accuracy.get_result()

    def __str__(self):
        position_acc, token_acc = self.get_result()
        return 'LineTokenEvaluator position_accuracy: {}, token_accuracy: {}'.format(position_acc, token_acc)

    def __repr__(self):
        return self.__str__()


class LineTokenSaver(Evaluator):
    def __init__(self, vocabulary, db_path, table_name, replace_table_name, ignore_token=None, end_id=None):
        self.vocabulary = vocabulary
        self.db_path = db_path
        self.table_name = table_name
        self.replace_table_name = replace_table_name
        create_table(self.db_path, self.table_name, self.replace_table_name)
        self.ignore_token = ignore_token
        self.end_id = end_id
        self.total_count = 0

    def filter_token_ids(self, token_ids, end):
        try:
            end_position = token_ids.index(end)
            token_ids = token_ids[:end_position]
        except ValueError as e:
            end_position = None
        return token_ids, end_position

    def save_records(self, params):
        run_sql_statment(self.db_path, self.table_name, 'insert_ignore', params, replace_table_name=self.replace_table_name)

    def add_result(self, output_ids, model_output, model_target, model_input, batch_data):
        # unfinished
        batch_size = model_output[0].size(0)
        self.total_count += model_output[0].size(0)
        ids = batch_data['id']
        codes = batch_data['code']
        position_line = torch.squeeze(torch.topk(F.softmax(model_output[0], dim=-1), dim=-1, k=1)[1], dim=-1).tolist()
        output_ids = output_ids.tolist()
        output_ids = [self.filter_token_ids(one, self.end_id) for one in output_ids]
        output_tokens = [[self.vocabulary.id_to_word(o) for o in one_lines] for one_lines, i in output_ids]
        output_lines = [json.dumps(one) for one in output_tokens]
        params = [one for one in zip(ids, codes, position_line, output_lines)]
        self.save_records(params)
        return 'LineTokenSaver saves {} code, total {} codes.'.format(batch_size, self.total_count)

    def clear_result(self):
        self.total_count = 0

    def get_result(self):
        return self.total_count

    def __str__(self):
        return 'LineTokenSaver saves {} code totally.'.format(self.total_count)

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    pass
