文件目录
SemanticsProgramFix
	config.py ---- 各个机器的静态路径配置，不要上传到git
	config_template ---- 静态路径配置模板，根据机器实际路径修改
	parameters_config.py ---- 实验配置，一个函数是一个实验配置，需要读取数据集、字典等，配置实验的各个参数，返回值为一个字典。此处需要添加或修改项目配置
	train.py ---- 训练脚本入口，实现了多个训练和测试的控制逻辑

	model/ ---- 模型目录。其中每个模型都需要实现create_loss_fn、create_parse_input_batch_data_fn、create_parse_target_batch_data_fn、expand_output_and_target_fn、multi_step_print_output_records_fn等一系列函数，放在模型文件内。
	read_data/ ---- 数据读取目录，将数据从数据库中读入并转为pandas.dataframe
		read_data_from_db.py ---- 读取数据库
		read_filter_data.py ---- 读取过滤后的数据。（限制长度、无法tokenize、去除相同题目和人的条目）
		read_experiment_data.py ---- 将数据划分为数据集（通常是3个：训练集、验证集、测试集）
	experiment/ ---- 读取数据并转化为实验所需格式的工具包（生成可迭代的数据集，生成字典，处理数据为模型输入的格式）
		experiment_dataset.py ---- 实验的数据集
		experiment_util.py ---- 数据库数据转化为数据集需求的输入格式
		parse_xy_util.py ---- 具体转换数据的逻辑
		load_data_vocabulary.py ---- 读取数据字典
	trained_model/ ---- 训练模型存储的位置，不要上传到git
	vocabulary/ ---- 通用字典
	database/ ---- 数据存储相关工具
	common/ ---- 通用工具包
		constants.py ---- 通用的常量存储
		util.py ---- python通用函数
		torch_util.py ---- torch通用函数
		evaluate_util.py ---- 测试通用对象
	scripts/ ---- 单独运行的小脚本
	data/ ---- 数据存储的目录，不要上传到git

对一个新的模型而言，需要完成read_data目录中读取数据的相关函数，完成model目录中实际的模型，完成experiment目录中将数据库数据转换为模型需要数据格式和数据集和字典，完成parameters_config.py中实验的配置。其中common/目录下的内容根据需求进行调用和新增或修改。


