import sys

from python_execution_trace.execution_trace.record import record

t = 0

def add_two(b):
    b = b + 2
    return b

@record(1)
def add_one(a):
    global t
    b = a + 1
    c = a + 2
    d = b + 3
    t = 5
    b = add_two(b)
    return b


if __name__ == '__main__':
    add_one(1)