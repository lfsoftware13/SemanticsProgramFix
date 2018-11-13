import re



def replace_content(content: str):
    content = re.sub('\$DIR/\.\./\.\./\.\./bin/genprog_tests\.py', 'python $DIR/../../../bin/genprog_tests.py', content)
    content = re.sub('\$DIR/\.\./\.\./tests/whitebox/[0-9]+\.out', '$DIR/../../tests/checksum', content)
    return content


def read_script_file():
    pass
