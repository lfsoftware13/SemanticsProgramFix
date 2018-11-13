import os
import sys

t = 1

def add_one(a):
    global oo
    print('after global')
    oo = 1
    print('in call begin')
    b = a + 1
    print('one line')
    c = b + 1
    print('before return')
    d = t + 1
    return b


def tracer_f(frame, event, arg):
    file_full_name = frame.f_code.co_filename
    # print('file full name: ', file_full_name)
    # print('file name: ', os.path.split(file_full_name)[1])
    if os.path is None or os.path.split(file_full_name)[1] != 'tracer_test.py':
        return tracer_f
    print_code_object(frame.f_code)
    print('f_locals: ', frame.f_locals)
    # print('f_globals: ', frame.f_globals)
    print('event', event)
    print('arg', arg)
    return tracer_f


def print_code_object(code_object):
    print('f_code: ', code_object)
    print('code_object.co_name: ', code_object.co_name)
    print('code_object.co_argcount: ', code_object.co_argcount)
    print('code_object.co_nlocals: ', code_object.co_nlocals)
    print('code_object.co_varnames: ', code_object.co_varnames)
    print('code_object.co_filename: ', code_object.co_filename)


def main():
    sys.settrace(tracer_f)
    # import tests.program
    print('before function')
    add_one(1)
    print('outer function')




if __name__ == '__main__':
    main()