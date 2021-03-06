import os
import keyword

from common.util import reverse_dict
from config import root, scrapyOJ_path, cache_path, temp_code_write_path


# scrapyOJ db path. all OJ data
scrapyOJ_DB_PATH = scrapyOJ_path


# project dir path
ROOT_PATH = root
DATA_PATH = os.path.join(ROOT_PATH, 'data')
CACHE_DATA_PATH = cache_path
TMP_FILE_PATH = os.path.join(ROOT_PATH, 'tmp')
COMPILE_TMP_PATH = temp_code_write_path


# db path
TRAIN_DATA_DBPATH = os.path.join(DATA_PATH, 'train_data.db')
COMPILE_SUCCESS_DATA_DBPATH = os.path.join(DATA_PATH, 'compile_success_data.db')
FAKE_C_COMPILE_ERROR_DATA_DBPATH = os.path.join(DATA_PATH, 'fake_c_compile_error_data.db')

# table name
ACTUAL_C_ERROR_RECORDS = 'actual_c_error_records'
CPP_TESTCASE_ERROR_RECORDS = 'cpp_testcase_error_records'
C_COMPILE_SUCCESS_RECORDS = 'c_compile_success_records'
COMMON_C_ERROR_RECORDS = 'common_c_error_records'
RANDOM_C_ERROR_RECORDS = 'random_c_error_records'
SLK_SAMPLE_COMMON_C_ERROR_RECORDS_BASENAME = 'slk_sample_common_c_error_records'
SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TRAIN = 'slk_sample_common_c_error_records_train'
SLK_SAMPLE_COMMON_C_ERROR_RECORDS_VALID = 'slk_sample_common_c_error_records_valid'
SLK_SAMPLE_COMMON_C_ERROR_RECORDS_TEST = 'slk_sample_common_c_error_records_test'
COMMON_DEEPFIX_ERROR_RECORDS = 'common_deepfix_error_records'
DATA_RECORDS_DEEPFIX = 'data_records_deepfix'
ARTIFICIAL_CODE = 'artificalCode'

PYTHON_SUBMIT_TABLE = 'python_submit_table'
C_SUBMIT_TABLE = 'c_submit_table'
LINE_TOKEN_SAMPLE_RECORDS = 'line_token_sample_records_table'

# code status and language transform dict
verdict = {'OK': 1, 'REJECTED': 2, 'WRONG_ANSWER': 3, 'RUNTIME_ERROR': 4, 'TIME_LIMIT_EXCEEDED': 5, 'MEMORY_LIMIT_EXCEEDED': 6,
           'COMPILATION_ERROR': 7, 'CHALLENGED': 8, 'FAILED': 9, 'PARTIAL': 10, 'PRESENTATION_ERROR': 11, 'IDLENESS_LIMIT_EXCEEDED': 12,
           'SECURITY_VIOLATED': 13, 'CRASHED': 14, 'INPUT_PREPARATION_CRASHED': 15, 'SKIPPED': 16, 'TESTING': 17, 'SUBMITTED': 18}
langdict = {'GNU C': 1, 'GNU C11': 2, 'GNU C++': 3, 'GNU C++11': 4, 'GNU C++14': 5,
            'MS C++': 6, 'Mono C#': 7, 'MS C#': 8, 'D': 9, 'Go': 10,
            'Haskell': 11, 'Java 8': 12, 'Kotlin': 13, 'Ocaml': 14, 'Delphi': 15,
            'FPC': 16, 'Perl': 17, 'PHP': 18, 'Python 2': 19, 'Python 3': 20,
            'PyPy 2': 21, 'PyPy 3': 22, 'Ruby': 23, 'Rust': 24, 'Scala': 25,
            'JavaScript': 26}


# ---------------------------------------- pre diefined c token --------------------------------#

c_standard_library_defined_identifier = {
    'erfc',
    'nexttoward',
    'fflush',
    'lrintf',
    'rename',
    'nearbyintf',
    'atanhl',
    'logl',
    'vsprintf',
    'strncmp',
    'cosh',
    'truncf',
    'ldexpf',
    'putchar',
    'erf',
    'malloc',
    'puts',
    'powf',
    'roundl',
    'atan',
    'strcpy',
    'roundf',
    'round',
    'fprintf',
    'modff',
    'floor',
    'logbf',
    'freopen',
    'nextafter',
    'fmin',
    'strtof',
    'memset',
    'nexttowardf',
    'fputs',
    'putc',
    'ftell',
    'llrintl',
    'lround',
    'acoshl',
    'sqrtf',
    'stdin',
    'wcstombs',
    'atanf',
    'nextafterl',
    'fgets',
    'strxfrm',
    'erfcl',
    'strpbrk',
    'fread',
    'clearerr',
    'scalblnf',
    'abs',
    'atan2f',
    'exp2',
    'atol',
    'atan2',
    'strtol',
    'rint',
    'floorf',
    'atoi',
    'rintl',
    'fminl',
    'lgammaf',
    'logbl',
    'lgamma',
    'fmax',
    'scalblnl',
    'wctomb',
    'strrchr',
    'erff',
    'vsscanf',
    'nearbyint',
    'ilogbf',
    'fabsl',
    'sqrt',
    'scanf',
    'scalbln',
    'tgammal',
    'sscanf',
    'srand',
    'strcspn',
    'nexttowardl',
    'acosf',
    'log10l',
    'vprintf',
    'memcpy',
    'calloc',
    'expm1f',
    'labs',
    'scalbn',
    'strspn',
    'atof',
    'frexp',
    'logf',
    'cos',
    'rintf',
    'nanf',
    'vsnprintf',
    'ldexp',
    'qsort',
    'cosf',
    '__codecvt_noconv',
    'expm1l',
    'ferror',
    'log2',
    'truncl',
    'sinf',
    'nextafterf',
    'lroundl',
    'copysignl',
    'gets',
    'acosl',
    'tanf',
    'fopen',
    'fclose',
    'remainderf',
    'log2f',
    'strcoll',
    'fwrite',
    'strerror',
    'exp2f',
    'tanhf',
    'copysignf',
    'llround',
    'mbtowc',
    'feof',
    'fmodl',
    'remquo',
    'llroundf',
    'lgammal',
    'exp2l',
    'erfcf',
    'fscanf',
    'strstr',
    'tanhl',
    'sprintf',
    'sinh',
    'tgammaf',
    'asinl',
    'strtoll',
    'realloc',
    'llrintf',
    'strncat',
    'nan',
    'ceilf',
    'frexpl',
    'setbuf',
    'fma',
    'tanh',
    'getc',
    'atanhf',
    'abort',
    'getchar',
    'fsetpos',
    'copysign',
    'sinhl',
    'asinf',
    'nanl',
    'llrint',
    'memcmp',
    'atan2l',
    'strchr',
    'log',
    'trunc',
    'ilogb',
    'log2l',
    'hypot',
    'system',
    'ceil',
    'printf',
    'acos',
    'vfprintf',
    'scalbnl',
    'strtok',
    'logb',
    'lrint',
    'sqrtl',
    'exp',
    'nearbyintl',
    'atexit',
    'rand',
    'fseek',
    'atanl',
    'setvbuf',
    'strcmp',
    'fmod',
    'remove',
    'free',
    'acoshf',
    'erfl',
    'tmpnam',
    'bsearch',
    'remquof',
    'tanl',
    'coshl',
    'fdim',
    'atanh',
    'asin',
    'pow',
    'expm1',
    'strncpy',
    'fdimf',
    'tmpfile',
    'strtod',
    'tan',
    'fmaxf',
    'cosl',
    'llroundl',
    'asinh',
    'fmaxl',
    'fabs',
    'vfscanf',
    'strtoul',
    'log1pl',
    'memchr',
    'remainder',
    'stdout',
    'hypotf',
    'sinhf',
    'mbstowcs',
    'vscanf',
    'asinhf',
    'strlen',
    'expf',
    'strtoull',
    'remquol',
    'ilogbl',
    'perror',
    'cbrtf',
    'fdiml',
    'log10',
    'ldiv',
    'ungetc',
    'log10f',
    'cbrt',
    'sinl',
    'rewind',
    'fmaf',
    'frexpf',
    'acosh',
    'asinhl',
    'fmodf',
    'strcat',
    'div',
    'remainderl',
    'log1p',
    'floorl',
    'modfl',
    'hypotl',
    'fmal',
    'fputc',
    'fminf',
    'exit',
    'fabsf',
    'sin',
    'powl',
    'atoll',
    'mblen',
    'modf',
    'expl',
    'strtold',
    'cbrtl',
    'memmove',
    'stderr',
    'snprintf',
    'llabs',
    'lldiv',
    'lroundf',
    'log1pf',
    'tgamma',
    'coshf',
    'getenv',
    'fgetpos',
    'ceill',
    'lrintl',
    'fgetc',
    'scalbnf'
}

c_standard_library_defined_types = {
    '_IO_FILE',
    '__mbstate_t',
    'FILE',
    'float_t',
    '__off_t',
    '__ssize_t',
    'lldiv_t',
    '__compar_fn_t',
    'wchar_t',
    'double_t',
    '__gnuc_va_list',
    '__off64_t',
    'fpos_t',
    '_G_fpos_t',
    'size_t',
    '_IO_lock_t',
    'ldiv_t',
    'div_t'
}

c_keywords = (
    '_BOOL', '_COMPLEX', 'AUTO', 'BREAK', 'CASE', 'CHAR', 'CONST',
    'CONTINUE', 'DEFAULT', 'DO', 'DOUBLE', 'ELSE', 'ENUM', 'EXTERN',
    'FLOAT', 'FOR', 'GOTO', 'IF', 'INLINE', 'INT', 'LONG',
    'REGISTER', 'OFFSETOF',
    'RESTRICT', 'RETURN', 'SHORT', 'SIGNED', 'SIZEOF', 'STATIC', 'STRUCT',
    'SWITCH', 'TYPEDEF', 'UNION', 'UNSIGNED', 'VOID',
    'VOLATILE', 'WHILE', '__INT128',
)

c_keyword_map = {}
for c_kwd in c_keywords:
    if c_kwd == '_BOOL':
        c_keyword_map['_Bool'] = c_kwd
    elif c_kwd == '_COMPLEX':
        c_keyword_map['_Complex'] = c_kwd
    else:
        c_keyword_map[c_kwd.lower()] = c_kwd

c_keyword_map = reverse_dict(c_keyword_map)

c_operator_map = {
    'PLUS': '+',
    'MINUS': '-',
    'TIMES': '*',
    'DIVIDE': '/',
    'MOD': '%',
    'OR': '|',
    'AND': '&',
    'NOT': '~',
    'XOR': '^',
    'LSHIFT': '<<',
    'RSHIFT': '>>',
    'LOR': '||',
    'LAND': '&&',
    'LNOT': '!',
    'LT': '<',
    'GT': '>',
    'LE': '<=',
    'GE': '>=',
    'EQ': '==',
    'NE': '!=',

    # Assignment operators
    'EQUALS': '=',
    'TIMESEQUAL': '*=',
    'DIVEQUAL': '/=',
    'MODEQUAL': '%=',
    'PLUSEQUAL': '+=',
    'MINUSEQUAL': '-=',
    'LSHIFTEQUAL': '<<=',
    'RSHIFTEQUAL': '>>=',
    'ANDEQUAL': '&=',
    'OREQUAL': '|=',
    'XOREQUAL': '^=',

    # Increment/decrement
    'PLUSPLUS': '++',
    'MINUSMINUS': '--',

    # ->
    'ARROW': '->',

    # ?
    'CONDOP': '?',

    # Delimeters
    'LPAREN': '(',
    'RPAREN': ')',
    'LBRACKET': '[',
    'RBRACKET': ']',
    'COMMA': ',',
    'PERIOD': '.',
    'SEMI': ';',
    'COLON': ':',
    'ELLIPSIS': '...',

    'LBRACE': '{',
    'RBRACE': '}',
}

pre_defined_c_tokens = set(c_keyword_map.values()) | set(c_operator_map.values())
pre_defined_c_tokens_map = {**c_keyword_map, **c_operator_map}
pre_defined_c_label = set(pre_defined_c_tokens_map.keys())

pre_defined_c_library_tokens = c_standard_library_defined_identifier | c_standard_library_defined_types

pre_defined_py_keyword = set(keyword.kwlist)
pre_defined_py_operator = {
    # Arithmetic Operators
    '+',
    '-',
    '*',
    '/',
    '%',
    '**',
    '//',
    '',

    # Comparison Operators
    '==',
    '!=',
    '<>',
    '>',
    '<',
    '>=',
    '<=',

    # Assignment Operators
    '=',
    '+=',
    '-=',
    '*=',
    '/=',
    '%=',
    '**=',
    '//=',

    # Bitwise Operators
    '&',
    '|',
    '^',
    '~',
    '<<',
    '>>',

    # Logical Operators
    'and',
    'or',
    'not',

    # Membership Operators
    'in',
    'not in',

    # Identity Operators
    'is',
    'is not',
}

pre_defined_py_label = pre_defined_py_keyword | pre_defined_py_operator
