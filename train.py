import os
import random

import torch
import pandas as pd

from torch import nn, optim
from tqdm import tqdm

from common import torch_util, problem_util, util
from common.constants import DATA_RECORDS_DEEPFIX
from common.evaluate_util import CompileResultEvaluate
from common.logger import init_a_file_logger, info
from common.problem_util import to_cuda
from common.util import data_loader, compile_code_ids_list, add_pid_to_file_path, save_addition_data, \
    create_special_tokens_ids, create_special_token_mask_list
import torch.functional as F

from database.database_util import create_table, insert_items, run_sql_statment

IGNORE_TOKEN = -1


def get_model(model_fn, model_params, path, load_previous=False, parallel=False, gpu_index=None):
    m = model_fn(
        **model_params
    )
    # to_cuda(m)
    if parallel:
        m = nn.DataParallel(m.cuda(), device_ids=[0, 1])
    elif gpu_index is not None:
        m = nn.DataParallel(m.cuda(gpu_index), device_ids=[gpu_index])
    else:
        m = nn.DataParallel(m.cuda(), device_ids=[0])
    if load_previous:
        # torch_util.load_model(m, path, map_location={'cuda:0': 'cuda:1'})
        torch_util.load_model(m, path)
        print("load previous model from {}".format(path))
    else:
        print("create new model")
    if gpu_index is None and not parallel:
        m = m.module.cpu()
    return m


def train(model, dataset, batch_size, loss_function, optimizer, clip_norm, epoch_ratio, parse_input_batch_data_fn,
          parse_target_batch_data_fn, create_output_ids_fn, evaluate_obj_list):
    total_loss = to_cuda(torch.Tensor([0]))
    steps = 0
    for o in evaluate_object_list:
        o.clear_result()
    model.train()

    with tqdm(total=(len(dataset)*epoch_ratio)) as pbar:
        for batch_data in data_loader(dataset, batch_size=batch_size, is_shuffle=True, drop_last=True, epoch_ratio=epoch_ratio):
            model.zero_grad()

            model_input = parse_input_batch_data_fn(batch_data, do_sample=False)
            model_output = model.forward(*model_input)

            model_target = parse_target_batch_data_fn(batch_data)
            loss = loss_function(*model_output, *model_target)

            loss.backward()
            optimizer.step()

            output_ids = create_output_ids_fn(model_output, model_input, False)
            for evaluator in evaluate_obj_list:
                evaluator.add_result(output_ids, model_output, model_target, model_input, batch_data=batch_data)

            total_loss += loss.data

            step_output = 'in train step {}  loss: {}'.format(steps, loss.data.item())
            # print(step_output)
            info(step_output)

            steps += 1
            pbar.update(batch_size)

    return evaluate_obj_list, (total_loss / steps).item()


def evaluate(model, dataset, batch_size, loss_function, parse_input_batch_data_fn, parse_target_batch_data_fn,
             do_sample=False, print_output=False, print_output_fn=None, create_output_ids_fn=None, evaluate_obj_list=[],
             expand_output_and_target_fn=None):
    total_loss = to_cuda(torch.Tensor([0]))
    loss = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    steps = 0
    for o in evaluate_object_list:
        o.clear_result()
    model.eval()

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=True):
                model.zero_grad()

                # model_input = parse_input_batch_data(batch_data)
                model_input = parse_input_batch_data_fn(batch_data, do_sample=do_sample)
                # model_output = model.forward(*model_input, test=do_sample)
                if do_sample:
                    model_output = model.forward(*model_input, do_sample=True)

                    model_target = parse_target_batch_data_fn(batch_data)
                    if model_target is not None:
                        model_output, model_target = expand_output_and_target_fn(model_output, model_target)
                else:
                    model_output = model.forward(*model_input)
                    model_target = parse_target_batch_data_fn(batch_data)

                output_ids = create_output_ids_fn(model_output, model_input, do_sample)
                total_batch += batch_size

                if model_target is not None:
                    loss = loss_function(*model_output, *model_target)
                    total_loss += loss.data

                step_output = 'in evaluate step {}  loss: {}, '.format(steps, loss.data.item())
                for evaluator in evaluate_obj_list:
                    res = evaluator.add_result(output_ids, model_output, model_target, model_input, batch_data=batch_data)
                    step_output += res
                # print(step_output)
                info(step_output)

                if print_output and steps % 10 == 0:
                    print_output_fn(final_output=output_ids, model_output=model_output, model_target=model_target,
                                    model_input=model_input, batch_data=batch_data, step_i=steps)

                steps += 1
                pbar.update(batch_size)

    return evaluate_obj_list, (total_loss / steps).item()


def multi_step_evaluate(model, dataset, batch_size, parse_input_batch_data_fn, parse_target_batch_data_fn,
             do_sample=False, print_output=False, create_output_ids_fn=None, evaluate_obj_list=[],
             expand_output_and_target_fn=None, max_step_times=0, vocabulary=None, file_path='',
                        create_multi_step_next_input_batch_fn=None, extract_includes_fn=lambda x: x['includes'],
                        print_output_fn=None, do_beam_search=False, target_file_path='main.out', do_save_data=False,
                        max_save_distance=None, save_records_to_database=False,
                        db_path='', table_name='', change_output_records_to_batch_fn=None, create_save_database_records_fn=None):
    total_loss = to_cuda(torch.Tensor([0]))
    total_batch = to_cuda(torch.Tensor([0]))
    steps = 0
    compile_evaluator = CompileResultEvaluate()
    compile_evaluator.clear_result()
    for o in evaluate_object_list:
        o.clear_result()

    model.eval()

    from common.pycparser_util import tokenize_by_clex_fn
    tokenize_fn = tokenize_by_clex_fn()

    # file_path = add_pid_to_file_path(file_path)
    # target_file_path = add_pid_to_file_path(target_file_path)

    with tqdm(total=len(dataset)) as pbar:
        with torch.no_grad():
            for batch_data in data_loader(dataset, batch_size=batch_size, drop_last=False):
                model.zero_grad()

                input_data = batch_data.copy()
                final_output_list = []
                output_records_list = []
                continue_list = [True for _ in range(batch_size)]
                result_list = [False for _ in range(batch_size)]
                result_records_list = []
                sample_steps = [-1 for _ in range(batch_size)]

                for i in range(max_step_times):
                    model_input = parse_input_batch_data_fn(input_data, do_sample=True)

                    model_output = model.forward(*model_input, do_sample=True, do_beam_search=do_beam_search)

                    input_data, final_output, output_records, final_output_name_list, continue_list = create_multi_step_next_input_batch_fn(input_data,
                                                                                                     model_input,
                                                                                                     model_output,
                                                                                                     continue_list,
                                                                                                     do_beam_search)
                    final_output_list += [final_output]
                    output_records_list += [output_records]

                    continue_list, result_list = compile_code_ids_list(final_output_name_list, continue_list, result_list, vocabulary=vocabulary,
                                                          includes_list=extract_includes_fn(input_data), file_path=file_path,
                                                                       target_file_path=target_file_path, do_compile_pool=True,
                                                                       need_transform=False)
                    sample_steps = [i+1 if s == -1 and not c else s for s, c in zip(sample_steps, continue_list)]

                    result_records_list += [result_list]
                    if sum(continue_list) == 0:
                        break
                sample_steps = [max_step_times if s == -1 else s for s in sample_steps]

                step_output = 'in evaluate step {}: '.format(steps)
                res = compile_evaluator.add_result(result_list)
                step_output += res
                for evaluator in evaluate_obj_list:
                    # customer evaluator interface
                    res = evaluator.add_result(result_list, batch_data=batch_data)
                    step_output += res
                # print(step_output)
                info(step_output)

                if print_output and steps % 1 == 0:
                    print_output_fn(output_records=output_records_list, final_output=final_output_list, batch_data=batch_data,
                                    step_i=steps, vocabulary=vocabulary, compile_result_list=result_records_list)


                steps += 1
                pbar.update(batch_size)
    evaluate_obj_list = [compile_evaluator] + evaluate_obj_list

    t_loss = (total_loss / steps).item() if steps != 0 else 0
    return evaluate_obj_list, t_loss


def train_and_evaluate(model, batch_size, train_dataset, valid_dataset, test_dataset,
                       learning_rate, epoches, saved_name, train_loss_fn, optimizer, optimizer_dict,
                       parse_input_batch_data_fn, parse_target_batch_data_fn,
                       create_output_ids_fn, evaluate_obj_list,
                       load_previous=False, is_debug=False, epoch_ratio=1.0, clip_norm=1,
                       do_sample_evaluate=False, do_sample=False, print_output=False, print_output_fn=None, expand_output_and_target_fn=None,
                       start_epoch=0, db_path=None, table_basename=None,
                       max_step_times=1, compile_file_path=None, do_multi_step_sample_evaluate=False,
                       create_multi_step_next_input_batch_fn=None, extract_includes_fn=None,
                       multi_step_sample_evaluator=[], vocabulary=None,
                       do_beam_search=False, target_file_path='main.out',
                       do_save_records_to_database=False, change_output_records_to_batch_fn=None,
                       create_save_database_records_fn=None, just_evaluate=False):
    valid_loss = 0
    test_loss = 0
    valid_accuracy = 0
    test_accuracy = 0
    valid_correct = 0
    test_correct = 0
    sample_valid_loss = 0
    sample_test_loss = 0
    sample_valid_accuracy = 0
    sample_test_accuracy = 0
    sample_valid_correct = 0
    sample_test_correct = 0

    save_path = os.path.join(config.save_model_root, saved_name)

    addition_dataset = None

    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, **optimizer_dict)

    if load_previous or just_evaluate:
        # valid_loss, valid_accuracy, valid_correct = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
        #                                       loss_function=loss_fn, vocab=vocabulary, add_value_mask=add_value_mask)
        # test_evaluator, test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
        #                                      loss_function=train_loss_fn, do_sample=False,
        #                                      parse_input_batch_data_fn=parse_input_batch_data_fn,
        #                                      parse_target_batch_data_fn=parse_target_batch_data_fn,
        #                                      create_output_ids_fn=create_output_ids_fn,
        #                                      evaluate_obj_list=evaluate_obj_list,
        #                                      expand_output_and_target_fn=expand_output_and_target_fn)
        # print('previous test loss: {}, evaluator : '.format(test_loss))
        # for evaluator in test_evaluator:
        #     print(evaluator)

        if do_multi_step_sample_evaluate:
            multi_step_test_evalutor, sample_test_loss = multi_step_evaluate(model=model, dataset=test_dataset,
                                                              batch_size=batch_size, do_sample=True,
                                                              print_output=print_output,
                                                              parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                              parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                              create_output_ids_fn=create_output_ids_fn,
                                                              evaluate_obj_list=multi_step_sample_evaluator,
                                                              expand_output_and_target_fn=expand_output_and_target_fn,
                                                             extract_includes_fn=extract_includes_fn,
                                                             vocabulary=vocabulary, file_path=compile_file_path,
                                                                             max_step_times=max_step_times,
                                                                             create_multi_step_next_input_batch_fn=create_multi_step_next_input_batch_fn,
                                                                             print_output_fn=print_output_fn,
                                                                             do_beam_search=do_beam_search,
                                                                             target_file_path=target_file_path,
                                                                                do_save_data=False,
                                                                                save_records_to_database=do_save_records_to_database,
                                                                                db_path=db_path, table_name=table_basename,
                                                                                change_output_records_to_batch_fn=change_output_records_to_batch_fn,
                                                                                create_save_database_records_fn=create_save_database_records_fn)
            print('previous sample test loss: {}, evaluator : '.format(sample_test_loss))
            info('previous sample test loss: {}, evaluator : '.format(sample_test_loss))
            for evaluator in multi_step_test_evalutor:
                print(evaluator)
                info(evaluator)
            return

        if do_sample_evaluate:
            sample_test_evalutor, sample_test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                                              loss_function=train_loss_fn, do_sample=do_sample,
                                                              print_output=print_output, print_output_fn=print_output_fn,
                                                              parse_input_batch_data_fn=parse_input_batch_data_fn,
                                                              parse_target_batch_data_fn=parse_target_batch_data_fn,
                                                              create_output_ids_fn=create_output_ids_fn,
                                                              evaluate_obj_list=evaluate_obj_list,
                                                              expand_output_and_target_fn=expand_output_and_target_fn)
            print('previous sample test loss: {}, evaluator : '.format(sample_test_loss))
            info('previous sample test loss: {}, evaluator : '.format(sample_test_loss))
            for evaluator in sample_test_evalutor:
                print(evaluator)
                info(evaluator)
            return
        evaluate_output = 'evaluate: valid loss of {}, test loss of {}, ' \
                          'valid_accuracy result of {}, test_accuracy result of {}, ' \
                          'valid correct result of {}, test correct result of {}, ' \
                          'sample valid loss: {}, sample test loss: {}, ' \
                          'sample valid accuracy: {}, sample test accuracy: {}, ' \
                          'sample valid correct: {}, sample test correct: {}'.format(
            valid_loss, test_loss, valid_accuracy, test_accuracy, valid_correct, test_correct,
            sample_valid_loss, sample_test_loss, sample_valid_accuracy, sample_test_accuracy, sample_valid_correct, sample_test_correct)
        print(evaluate_output)
        info(evaluate_output)

    combine_train_set = train_dataset
    for epoch in range(start_epoch, start_epoch+epoches):
        print('----------------------- in epoch {} --------------------'.format(epoch))

        train_evaluator, train_loss = train(model=model, dataset=combine_train_set, batch_size=batch_size,
                                            loss_function=train_loss_fn, optimizer=optimizer, clip_norm=clip_norm,
                                            epoch_ratio=epoch_ratio, parse_input_batch_data_fn=parse_input_batch_data_fn,
                                            parse_target_batch_data_fn=parse_target_batch_data_fn,
                                            create_output_ids_fn=create_output_ids_fn,
                                            evaluate_obj_list=evaluate_obj_list, )
        print('epoch: {} loss: {}, train evaluator : '.format(epoch, train_loss))
        info('epoch: {} loss: {}, train evaluator : '.format(epoch, train_loss))
        for evaluator in train_evaluator:
            print(evaluator)
            info(evaluator)

        valid_evaluator, valid_loss = evaluate(model=model, dataset=valid_dataset, batch_size=batch_size,
                                               loss_function=train_loss_fn,
                                               parse_input_batch_data_fn=parse_input_batch_data_fn,
                                               parse_target_batch_data_fn=parse_target_batch_data_fn,
                                               create_output_ids_fn=create_output_ids_fn,
                                               evaluate_obj_list=evaluate_obj_list,
                                               expand_output_and_target_fn=expand_output_and_target_fn)
        print('epoch: {} loss: {}, valid evaluator : '.format(epoch, valid_loss))
        info('epoch: {} loss: {}, valid evaluator : '.format(epoch, valid_loss))
        for evaluator in valid_evaluator:
            print(evaluator)
            info(evaluator)
        test_evaluator, test_loss = evaluate(model=model, dataset=test_dataset, batch_size=batch_size,
                                             loss_function=train_loss_fn,
                                             parse_input_batch_data_fn=parse_input_batch_data_fn,
                                             parse_target_batch_data_fn=parse_target_batch_data_fn,
                                             create_output_ids_fn=create_output_ids_fn,
                                             evaluate_obj_list=evaluate_obj_list,
                                             expand_output_and_target_fn=expand_output_and_target_fn, )
        print('epoch: {} loss: {}, test evaluator : '.format(epoch, test_loss))
        info('epoch: {} loss: {}, test evaluator : '.format(epoch, test_loss))
        for evaluator in test_evaluator:
            print(evaluator)
            info(evaluator)

        if not is_debug:
            torch_util.save_model(model, save_path+str(epoch))


if __name__ == '__main__':
    import parameters_config
    import config
    import argparse

    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    random.seed(100)

    import sys

    sys.setrecursionlimit(5000)

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_previous", type=boolean_string, default=False)
    parser.add_argument("--debug", type=boolean_string, default=True)
    parser.add_argument("--config_name", type=str)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--parallel", type=boolean_string)
    parser.add_argument("--just_evaluate", type=boolean_string, default=False)
    parser.add_argument("--output_log", type=str, default=None)
    args = parser.parse_args()
    load_previous = args.load_previous
    problem_util.GPU_INDEX = args.gpu
    problem_util.Parallel = args.parallel
    is_debug = args.debug
    just_evaluate = args.just_evaluate

    p_config = parameters_config.__dict__.get(args.config_name)(is_debug)
    epoches = p_config.get("epcohes", 20)
    learning_rate = p_config.get("learning_rate", 20)
    batch_size = p_config.get("batch_size", 32)
    train_loss_fn = p_config.get("train_loss", nn.CrossEntropyLoss)
    clip_norm = p_config.get("clip_norm", 10)
    optimizer = p_config.get("optimizer", optim.SGD)
    optimizer_dict = p_config.get("optimizer_dict", dict())
    epoch_ratio = p_config.get("epoch_ratio", 0.25)
    start_epoch = p_config.get('start_epoch', 0)
    ac_copy_train = p_config.get('ac_copy_train', False)
    ac_copy_radio = p_config.get('ac_copy_radio', 0.2)
    evaluate_object_list = p_config.get("evaluate_object_list")
    do_sample_evaluate = p_config.get('do_sample_evaluate', False)
    do_sample = p_config.get('do_sample_evaluate', False)
    do_sample_and_save = p_config.get('do_sample_and_save', False)
    # label_preprocess_fn = p_config.get("label_preprocess", lambda x: to_cuda(torch.LongTensor(x['label'])))
    # scheduler_fn = p_config.get("scheduler_fn", lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=3, verbose=True))
    save_root_path = os.path.join(config.save_model_root, p_config.get("name"))
    util.make_dir(save_root_path)
    print("save_root_path:{}".format(save_root_path))
    init_a_file_logger(args.output_log)
    print("logger_file_path:{}".format(args.output_log))
    # need_pad = p_config.get("need_pad", False)
    save_name = p_config['save_name']
    parse_input_batch_data_fn = p_config['parse_input_batch_data_fn']
    parse_target_batch_data_fn = p_config['parse_target_batch_data_fn']
    expand_output_and_target_fn = p_config.get('expand_output_and_target_fn', None)
    db_path = p_config.get('db_path', None)
    table_basename = p_config.get('table_basename', None)
    do_save_records_to_database = p_config.get('do_save_records_to_database', False)
    create_output_ids_fn = p_config['create_output_ids_fn']
    vocabulary = p_config['vocabulary']

    load_addition_generate_iterate_solver_train_dataset_fn = p_config.get('load_addition_generate_iterate_solver_train_dataset_fn', None)
    addition_train = p_config.get('addition_train', False)

    do_multi_step_sample_evaluate = p_config['do_multi_step_sample_evaluate']
    max_step_times = p_config['max_step_times']
    create_multi_step_next_input_batch_fn = p_config['create_multi_step_next_input_batch_fn']
    compile_file_path = p_config['compile_file_path']
    target_file_path = p_config['target_file_path']
    extract_includes_fn = p_config['extract_includes_fn']
    print_output = p_config['print_output']
    print_output_fn = p_config['print_output_fn']

    do_beam_search = p_config.get('do_beam_search', False)

    random_embedding = p_config.get('random_embedding', False)
    use_ast = p_config.get('use_ast', False)

    change_output_records_to_batch_fn = p_config.get('change_output_records_to_batch_fn', None)
    create_save_database_records_fn = p_config.get('create_save_database_records_fn', None)

    model_path = os.path.join(save_root_path, p_config['load_model_name'])
    model = get_model(
        p_config['model_fn'],
        p_config['model_dict'],
        model_path,
        load_previous=load_previous,
        parallel=problem_util.Parallel,
        gpu_index=problem_util.GPU_INDEX,
    )

    train_data, val_data, test_data = p_config.get("data")
    if train_data is not None:
        print("The size of train data: {}".format(len(train_data)))
    if val_data is not None:
        print("The size of val data: {}".format(len(val_data)))
    if test_data is not None:
        print("The size of test data: {}".format(len(test_data)))
    train_and_evaluate(model=model, batch_size=batch_size, train_dataset=train_data, valid_dataset=val_data, test_dataset=test_data,
                       learning_rate=learning_rate, epoches=epoches, saved_name=save_name, train_loss_fn=train_loss_fn,
                       optimizer=optimizer, optimizer_dict=optimizer_dict,
                       parse_input_batch_data_fn=parse_input_batch_data_fn, parse_target_batch_data_fn=parse_target_batch_data_fn,
                       create_output_ids_fn=create_output_ids_fn, evaluate_obj_list=evaluate_object_list,
                       load_previous=load_previous, is_debug=is_debug, epoch_ratio=epoch_ratio, clip_norm=clip_norm, start_epoch=start_epoch,
                       do_sample_evaluate=do_sample_evaluate, do_sample=do_sample, print_output=print_output, print_output_fn=print_output_fn,
                       max_step_times=max_step_times, compile_file_path=compile_file_path, do_multi_step_sample_evaluate=do_multi_step_sample_evaluate,
                       expand_output_and_target_fn=expand_output_and_target_fn, db_path=db_path,
                       table_basename=table_basename, vocabulary=vocabulary,
                       create_multi_step_next_input_batch_fn=create_multi_step_next_input_batch_fn,
                       extract_includes_fn=extract_includes_fn,
                       do_beam_search=do_beam_search, target_file_path=target_file_path, do_save_records_to_database=do_save_records_to_database,
                       change_output_records_to_batch_fn=change_output_records_to_batch_fn,
                       create_save_database_records_fn=create_save_database_records_fn,
                       just_evaluate=just_evaluate,
                       )

    # test_loss, train_test_loss = evaluate(model, test_data, batch_size, evaluate_object_list,
    #                                       train_loss_fn, "test_evaluate", label_preprocess_fn)
    # print("train_test_loss is {}".format(train_test_loss.item(),))
    # for o in  test_loss:
    #     print(o)