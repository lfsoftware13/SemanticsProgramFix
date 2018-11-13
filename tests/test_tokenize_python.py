import unittest

from common.util import tokenize_python_code, tokenize_python_code_with_type


class TestTokenizePythonCode(unittest.TestCase):

    def setUp(self):
        self.code = r'''n, k = map(int, input().split())
l = list(map(int, input().split()))
i = 0
buf = 0
while i < n:
    if l[i] + buf > 8:
        k = k - 8
        buf = buf + l[i] - 8
    else:
        k = k - buf - l[i]
        buf = 0

    i += 1
    print('day (i) = ', i,)
    if k <= 0:
        print(i)
        break
if k > 0:
    print(-1)'''

    def test_tokenize(self):
        tokens = tokenize_python_code(self.code)
        print(len(tokens))
        print(tokens)
        self.assertEquals(len(tokens), 143)

    def test_tokenize_with_type(self):
        tokens = tokenize_python_code_with_type(self.code)
        print(len(tokens))
        print(tokens)
        self.assertEquals(len(tokens), 143)
