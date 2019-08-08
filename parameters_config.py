from common.constants import LINE_TOKEN_SAMPLE_RECORDS
from common.opt import OpenAIAdam


def example_config(is_debug):


    from experiment.load_data_vocabulary import create_deepfix_common_error_vocabulary
    from experiment.experiment_dataset import load_deepfix_sequence_dataset
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'],
                                                        unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    datasets = load_deepfix_sequence_dataset(is_debug, vocabulary=vocabulary, only_sample=False)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.simple_seq2seq import SimpleSeq2Seq
    from model.simple_seq2seq import create_parse_input_batch_data_fn
    from model.simple_seq2seq import create_parse_target_batch_data_fn
    from model.simple_seq2seq import expand_output_and_target_fn
    from model.simple_seq2seq import create_loss_fn
    from model.simple_seq2seq import multi_step_print_output_records_fn
    from model.simple_seq2seq import extract_includes_fn
    from model.simple_seq2seq import create_output_ids_fn
    return {
        'name': 'example_config',
        'save_name': 'example_config.pkl',
        'load_model_name': 'example_config.pkl',

        'model_fn': SimpleSeq2Seq,
        'model_dict':
            {'vocab_size': vocabulary.vocabulary_size, 'max_len': max_length, 'input_size': 400, 'hidden_size': 400,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0, 'n_layers': 3,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': False},

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(begin_id=begin_id, end_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': extract_includes_fn(),

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}


def line_token_config1(is_debug):


    from experiment.load_data_vocabulary import create_deepfix_python_semantics_common_error_vocabulary
    from experiment.experiment_dataset import load_deepfix_semantics_dataset
    vocabulary = create_deepfix_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                            unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])
    datasets = load_deepfix_semantics_dataset(is_debug, vocabulary=vocabulary, only_sample=False)

    batch_size = 16
    epoches = 80
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.rnn_one_token_model import create_parse_input_batch_data_fn
    from model.rnn_one_token_model import create_parse_target_batch_data_fn
    from model.rnn_one_token_model import expand_output_and_target_fn
    from model.rnn_one_token_model import create_loss_fn
    from model.rnn_one_token_model import multi_step_print_output_records_fn
    from model.rnn_one_token_model import create_output_ids_fn
    from model.rnn_one_token_model import LineRNNModel
    from common.evaluate_util import LineTokenEvaluator
    from model.rnn_one_token_model import print_output_fn
    return {
        'name': 'line_token_config1',
        'save_name': 'line_token_config1.pkl',
        'load_model_name': 'line_token_config1.pkl',

        'model_fn': LineRNNModel,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'input_size': 400, 'hidden_size': 400,
             'encoder_layer_nums': 3, 'decoder_layer_nums': 3, 'max_length': max_length,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': True},

        'do_sample_evaluate': True,
        'do_sample': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': True,
        'print_output_fn': print_output_fn(eos_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [LineTokenEvaluator(ignore_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}


def line_correction_baseline1(is_debug):
    '''
    just embedding code per line, predict the error line and decode the line
    :param is_debug:
    :return:
    '''

    from experiment.load_data_vocabulary import create_fake_python_semantics_common_error_vocabulary
    from experiment.experiment_dataset import load_fake_python_semantics_dataset
    from config import CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH
    vocabulary = create_fake_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                            unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])

    max_sample_length = 35
    batch_size = 16
    epoches = 60
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    use_ast = False
    table_name = 'line_correction_baseline1'
    db_path = CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH

    if use_ast:
        from experiment.load_data_vocabulary import load_python_parser_vocabulary
        vocabulary = load_python_parser_vocabulary(vocabulary)

    datasets = load_fake_python_semantics_dataset(is_debug, vocabulary=vocabulary, max_sample_length=max_sample_length,
                                                  use_ast=use_ast, only_sample=False)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.rnn_one_token_model import create_parse_input_batch_data_fn
    from model.rnn_one_token_model import create_parse_target_batch_data_fn
    from model.rnn_one_token_model import expand_output_and_target_fn
    from model.rnn_one_token_model import create_loss_fn
    from model.rnn_one_token_model import multi_step_print_output_records_fn
    from model.rnn_one_token_model import create_output_ids_fn
    from model.rnn_one_token_model import LineRNNModel
    from common.evaluate_util import LineTokenEvaluator
    from model.rnn_one_token_model import print_output_fn
    from common.evaluate_util import LineTokenSaver
    return {
        'name': 'line_correction_baseline1',
        'save_name': 'line_correction_baseline1.pkl',
        'load_model_name': 'line_correction_baseline1.pkl',

        'model_fn': LineRNNModel,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'input_size': 400, 'hidden_size': 400,
             'encoder_layer_nums': 3, 'decoder_layer_nums': 3, 'max_length': max_length, 'max_sample_length': 35,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': True,
             'graph_embedding': None,
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': False, 'do_embedding': False},
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             },

        'do_sample_evaluate': False,
        'do_sample': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': print_output_fn(eos_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id, use_ast=use_ast),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [LineTokenEvaluator(ignore_id)],
        # 'evaluate_object_list': [LineTokenSaver(vocabulary, db_path, LINE_TOKEN_SAMPLE_RECORDS, table_name, ignore_id, end_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}


def line_correction_model1(is_debug):
    '''
    ggnn + rnn with ast
    :param is_debug:
    :return:
    '''

    from experiment.load_data_vocabulary import create_fake_python_semantics_common_error_vocabulary
    from experiment.experiment_dataset import load_fake_python_semantics_dataset
    from config import CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH
    vocabulary = create_fake_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                            unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])

    max_sample_length = 35
    batch_size = 16
    epoches = 60
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    use_ast = True
    table_name = 'line_correction_model1'
    db_path = CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH

    if use_ast:
        from experiment.load_data_vocabulary import load_python_parser_vocabulary
        vocabulary = load_python_parser_vocabulary(vocabulary)

    datasets = load_fake_python_semantics_dataset(is_debug, vocabulary=vocabulary, max_sample_length=max_sample_length,
                                                  use_ast=use_ast, only_sample=False)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.rnn_one_token_model import create_parse_input_batch_data_fn
    from model.rnn_one_token_model import create_parse_target_batch_data_fn
    from model.rnn_one_token_model import expand_output_and_target_fn
    from model.rnn_one_token_model import create_loss_fn
    from model.rnn_one_token_model import multi_step_print_output_records_fn
    from model.rnn_one_token_model import create_output_ids_fn
    from model.rnn_one_token_model import LineRNNModel
    from common.evaluate_util import LineTokenEvaluator
    from model.rnn_one_token_model import print_output_fn
    from common.evaluate_util import LineTokenSaver
    return {
        'name': 'line_correction_model1',
        'save_name': 'line_correction_model1.pkl',
        'load_model_name': 'line_correction_model1.pkl',

        'model_fn': LineRNNModel,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'input_size': 400, 'hidden_size': 400,
             'encoder_layer_nums': 3, 'decoder_layer_nums': 3, 'max_length': max_length, 'max_sample_length': 35,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': True,
             'graph_embedding': 'mixed',
             'graph_parameter': {"rnn_parameter": {'vocab_size': vocabulary.vocabulary_size,
                                                   'max_len': max_length, 'input_size': 400,
                                                   'input_dropout_p': 0.2, 'dropout_p': 0.2,
                                                   'n_layers': 1, 'bidirectional': True, 'rnn_cell': 'gru',
                                                   'variable_lengths': False, 'embedding': None,
                                                   'update_embedding': False, 'do_embedding': False},
                                 "graph_type": "ggnn",
                                 "graph_itr": 3,
                                 "dropout_p": 0.2,
                                 "mask_ast_node_in_rnn": False
                                 },
             },

        'do_sample_evaluate': False,
        'do_sample': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': print_output_fn(eos_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id, use_ast=use_ast),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [LineTokenEvaluator(ignore_id)],
        # 'evaluate_object_list': [LineTokenSaver(vocabulary, db_path, LINE_TOKEN_SAMPLE_RECORDS, table_name, ignore_id, end_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}


def line_correction_model2(is_debug):
    '''
    ggnn with ast
    :param is_debug:
    :return:
    '''

    from experiment.load_data_vocabulary import create_fake_python_semantics_common_error_vocabulary
    from experiment.experiment_dataset import load_fake_python_semantics_dataset, load_codeforces_real_python_semantics_dataset
    vocabulary = create_fake_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                            unk_token='<UNK>', addition_tokens=['<PAD>'])
    from config import CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])

    max_sample_length = 35
    batch_size = 16
    epoches = 60
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    use_ast = True
    table_name = 'line_correction_model2_75'
    db_path = CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH

    if use_ast:
        from experiment.load_data_vocabulary import load_python_parser_vocabulary
        vocabulary = load_python_parser_vocabulary(vocabulary)

    # training dataset
    # datasets = load_fake_python_semantics_dataset(is_debug, vocabulary=vocabulary, max_sample_length=max_sample_length,
    #                                               use_ast=use_ast, only_sample=False)
    # real error dataset
    datasets = load_codeforces_real_python_semantics_dataset(is_debug, vocabulary=vocabulary,
                                                             max_sample_length=max_sample_length,
                                                             use_ast=use_ast, only_sample=True)
    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100


    from model.rnn_one_token_model import create_parse_input_batch_data_fn
    from model.rnn_one_token_model import create_parse_target_batch_data_fn
    from model.rnn_one_token_model import expand_output_and_target_fn
    from model.rnn_one_token_model import create_loss_fn
    from model.rnn_one_token_model import multi_step_print_output_records_fn
    from model.rnn_one_token_model import create_output_ids_fn
    from model.rnn_one_token_model import LineRNNModel
    from common.evaluate_util import LineTokenEvaluator, LineTokenSaver
    from model.rnn_one_token_model import print_output_fn
    return {
        'name': 'line_correction_model2',
        'save_name': 'line_correction_model2.pkl',
        'load_model_name': 'line_correction_model2.pkl',

        'model_fn': LineRNNModel,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'input_size': 400, 'hidden_size': 400,
             'encoder_layer_nums': 3, 'decoder_layer_nums': 3, 'max_length': max_length, 'max_sample_length': 35,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': True,
             'graph_embedding': 'ggnn',
             'graph_parameter': {"graph_type": "ggnn",
                                   "graph_itr": 3,
                                   "dropout_p": 0.2,
                                   },
             },

        'do_sample_evaluate': True,
        'do_sample': True,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': print_output_fn(eos_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id, use_ast=use_ast),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        # 'evaluate_object_list': [LineTokenEvaluator(ignore_id)],
        'evaluate_object_list': [LineTokenSaver(vocabulary, db_path, LINE_TOKEN_SAMPLE_RECORDS, table_name, ignore_id, end_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}


def line_correction_model3(is_debug):
    '''
    only rnn without ast
    :param is_debug:
    :return:
    '''

    from experiment.load_data_vocabulary import create_fake_python_semantics_common_error_vocabulary
    from experiment.experiment_dataset import load_fake_python_semantics_dataset, load_codeforces_real_python_semantics_dataset
    from config import CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH
    vocabulary = create_fake_python_semantics_common_error_vocabulary(begin_tokens=['<BEGIN>'], end_tokens=['<END>'],
                                                                            unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.word_to_id(vocabulary.begin_tokens[0])
    end_id = vocabulary.word_to_id(vocabulary.end_tokens[0])

    max_sample_length = 35
    batch_size = 16
    epoches = 60
    ignore_id = -1
    max_length = 500
    epoch_ratio = 1.0
    use_ast = False
    table_name = 'line_correction_model3'
    db_path = CODEFORCES_SEMANTIC_PYTHON_DATA_LINE_TOKEN_RECORDS_DBPATH

    if use_ast:
        from experiment.load_data_vocabulary import load_python_parser_vocabulary
        vocabulary = load_python_parser_vocabulary(vocabulary)

    datasets = load_fake_python_semantics_dataset(is_debug, vocabulary=vocabulary, max_sample_length=max_sample_length,
                                                  use_ast=use_ast, only_sample=False)
    # real error dataset
    # datasets = load_codeforces_real_python_semantics_dataset(is_debug, vocabulary=vocabulary,
    #                                                          max_sample_length=max_sample_length,
    #                                                          use_ast=use_ast, only_sample=True)

    train_len = len(datasets[0]) * epoch_ratio if datasets[0] is not None else 100

    from model.rnn_one_token_model import create_parse_input_batch_data_fn
    from model.rnn_one_token_model import create_parse_target_batch_data_fn
    from model.rnn_one_token_model import expand_output_and_target_fn
    from model.rnn_one_token_model import create_loss_fn
    from model.rnn_one_token_model import multi_step_print_output_records_fn
    from model.rnn_one_token_model import create_output_ids_fn
    from model.rnn_one_token_model import LineRNNModel
    from common.evaluate_util import LineTokenEvaluator
    from model.rnn_one_token_model import print_output_fn
    from common.evaluate_util import LineTokenSaver
    return {
        'name': 'line_correction_model3',
        'save_name': 'line_correction_model3.pkl',
        'load_model_name': 'line_correction_model3.pkl',

        'model_fn': LineRNNModel,
        'model_dict':
            {'vocabulary_size': vocabulary.vocabulary_size, 'input_size': 400, 'hidden_size': 400,
             'encoder_layer_nums': 3, 'decoder_layer_nums': 3, 'max_length': max_length, 'max_sample_length': 35,
             'begin_token': begin_id, 'end_token': end_id, 'input_dropout_p': 0, 'dropout_p': 0,
             'bidirectional': True, 'rnn_cell': 'gru', 'use_attention': True,
             'graph_embedding': 'rnn',
             'graph_parameter': {'vocab_size': vocabulary.vocabulary_size, 'max_len': max_length, 'input_size': 400,
                                 'input_dropout_p': 0.2, 'dropout_p': 0.2, 'n_layers': 3, 'bidirectional': True,
                                 'rnn_cell': 'gru', 'variable_lengths': False, 'embedding': None,
                                 'update_embedding': False, 'do_embedding': False}
             },

        'do_sample_evaluate': False,
        'do_sample': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': print_output_fn(eos_id=end_id, vocabulary=vocabulary),

        'extract_includes_fn': None,

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(ignore_id, use_ast=use_ast),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': create_output_ids_fn(end_id),
        'train_loss': create_loss_fn(ignore_id),
        'evaluate_object_list': [LineTokenEvaluator(ignore_id)],
        # 'evaluate_object_list': [LineTokenSaver(vocabulary, db_path, LINE_TOKEN_SAMPLE_RECORDS, table_name, ignore_id, end_id)],

        'epcohes': epoches,
        'start_epoch': 0,
        'epoch_ratio': epoch_ratio,
        'learning_rate': 6.25e-5,
        'batch_size': batch_size,
        'clip_norm': 1,
        'optimizer': OpenAIAdam,
        'optimizer_dict': {'schedule': 'warmup_linear', 'warmup': 0.002,
                           't_total': epoch_ratio * epoches * train_len//batch_size, 'max_grad_norm': 10},
        'data': datasets}

