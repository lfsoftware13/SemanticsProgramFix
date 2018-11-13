import ast


def ast_main():
    code = r'''
import os
def add(a, b):
    a = a + b
    return a
c = add(1, 2)
print(c)
    '''
    expr_ast = ast.parse(code)
    print(ast.dump(expr_ast))


if __name__ == '__main__':
    ast_main()