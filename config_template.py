import os

num_processes = 6

CODEFLAWS_BENCHMARK = r"D:\MachineLearning\dataset\codeflaws"
CODEFLAWS_BENCHMARK_df_target = r"data/codeflaws.pkl"

root = r"G:\Project\SemanticsProgramFix"
cache_path = os.path.join(root, 'data', 'cache_data')
scrapyOJ_path = r'G:\Project\program_ai\data\scrapyOJ.db'
temp_code_write_path = r'tmp'

python_db_path = r"G:\Project\SemanticsProgramFix\data\python_data.db"

FAKE_DEEPFIX_ERROR_DATA_DBPATH = os.path.join(root, 'data', 'fake_deepfix_error_data.db')