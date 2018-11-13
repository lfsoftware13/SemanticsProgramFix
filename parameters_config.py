from common.opt import OpenAIAdam


def example_config(is_debug):


    from experiment.load_data_vocabulary import create_deepfix_common_error_vocabulary
    vocabulary = create_deepfix_common_error_vocabulary(begin_tokens=['<BEGIN>', '<INNER_BEGIN>'],
                                                        end_tokens=['<END>', '<INNER_END>'],
                                                        unk_token='<UNK>', addition_tokens=['<PAD>'])
    begin_id = vocabulary.begin_tokens[0]
    end_id = vocabulary.end_tokens[0]
    datasets = None

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
    return {
        'name': 'graph_encoder_sample_config2',
        'save_name': 'graph_encoder_sample_config2.pkl',
        'load_model_name': 'graph_encoder_sample_config2.pkl',

        'model_fn': SimpleSeq2Seq,
        'model_dict':
            { },

        'do_sample_evaluate': False,

        'do_multi_step_sample_evaluate': False,
        'max_step_times': 10,
        'create_multi_step_next_input_batch_fn': None,
        'compile_file_path': '/dev/shm/main.c',
        'target_file_path': '/dev/shm/main.out',

        'multi_step_sample_evaluator': [],
        'print_output': False,
        'print_output_fn': multi_step_print_output_records_fn(begin_id=begin_id, end_id=end_id, vocabulary=vocabulary),

        'vocabulary': vocabulary,
        'parse_input_batch_data_fn': create_parse_input_batch_data_fn(),
        'parse_target_batch_data_fn': create_parse_target_batch_data_fn(ignore_id),
        'expand_output_and_target_fn': expand_output_and_target_fn(ignore_id),
        'create_output_ids_fn': None,
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
