import os
from datasets import Dataset

# ======================
# 数据层配置
# ======================

# 默认目录，可通过 cfg.dataset.base_dir 覆盖
DEFAULT_BASE_DIR = "./data/evaluation"

# ======================
# 数据集参数
# ======================
# token 上限
math_token = 8192  # 数学类
med_token = 1500   # 医学类

# batch 上限
math_batch = 5
med_batch = 15
other_batch = 10

# 数据集分类
math_datasets = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
med_datasets = ["medqa", "medmcqa", "pubmedqa", "clinical_knowledge", "college_biology", 
                "college_medicine", "medical_genetics", "professional_medicine", "anatomy", "entropy_bench"]

# ======================
# 自动生成 max_token_dataset
# ======================
max_token_dataset = {ds: math_token for ds in math_datasets}
max_token_dataset.update({ds: med_token for ds in med_datasets})

# ======================
# 自动生成 max_batch_size
# ======================
max_batch_size = {ds: math_batch for ds in math_datasets}
max_batch_size.update({ds: med_batch for ds in med_datasets})

# ======================
# 多选题集合
# ======================
is_multi_choice = set(med_datasets)

# ======================
# 数据集加载函数
# ======================
def detect_question_type(question: str, answer: str = None) -> str:
    """
    检测问题类型：选择题或解答题
    Args:
        question: 问题文本
        answer: 答案文本（可选）
    Returns:
        "multiple_choice" 或 "open_ended"
    """
    # 检查问题中是否包含选择题特征
    choice_indicators = [
        "A)", "B)", "C)", "D)", "E)", "F)",  # 英文选项
        "A.", "B.", "C.", "D.", "E.", "F.",  # 英文选项（点号）
        "A、", "B、", "C、", "D、", "E、", "F、",  # 中文选项
        "A．", "B．", "C．", "D．", "E．", "F．",  # 中文选项（中文句号）
        "Choose", "Select", "Which", "What is the correct",  # 英文选择题关键词
        "选择", "选出", "哪个", "哪项", "正确答案",  # 中文选择题关键词
    ]
    
    question_lower = question.lower()
    for indicator in choice_indicators:
        if indicator.lower() in question_lower:
            return "multiple_choice"
    
    # 检查答案是否像选择题答案
    if answer:
        answer_clean = answer.strip().upper()
        if answer_clean in ["A", "B", "C", "D", "E", "F"]:
            return "multiple_choice"
    
    return "open_ended"

def standardize_dataset_format(dataset, dataset_name: str = None) -> list:
    """
    将数据集标准化为统一格式：list[dict] with question, answer, type
    Args:
        dataset: HuggingFace Dataset 或 list
        dataset_name: 数据集名称（用于类型检测）
    Returns:
        list[dict]: 标准化后的数据集
    """
    # 转换为list格式
    if hasattr(dataset, 'to_list'):
        data_list = dataset.to_list()
    else:
        data_list = list(dataset)
    
    standardized_data = []
    
    for i, item in enumerate(data_list):
        # 确保item是字典格式
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item at index {i} is not a dictionary: {type(item)}")
        
        # 提取question和answer
        question = item.get('question', item.get('Question', item.get('prompt', '')))
        answer = item.get('answer', item.get('Answer', item.get('response', '')))
        
        if not question:
            raise ValueError(f"Dataset item at index {i} missing 'question' field: {item}")
        
        # 检测问题类型
        question_type = detect_question_type(question, answer)
        
        # 构建标准化条目
        standardized_item = {
            'question': question,
            'answer': answer or '',
            'type': question_type
        }
        
        # 保留原始数据中的其他字段
        for key, value in item.items():
            if key not in ['question', 'answer', 'type']:
                standardized_item[key] = value
        
        standardized_data.append(standardized_item)
    
    return standardized_data

def load_dataset_from_exploration(path:str,default_segmantation='I need to carefully'):
    """
    从探索数据集加载数据集
    Args:
        path: 探索数据集路径
        */dataset_name/generations_seed_merged.json
    Returns:
        dataset: list[dict] 标准化格式的数据集
    """
    dataset = Dataset.from_json(path)
    new_dataset =[]
    for item in dataset:
        for gen in item['responses']:
            new_dataset.append({
                'question': item['question'],
                'answer': item['answer'],
                'prompt': item['prompt']+gen.split(default_segmantation)[0]+"</EXPLORATION>"
            })
    dataset_name = path.split("/")[-2]
    return new_dataset, dataset_name

def load_dataset_by_name(name: str, base_dir: str = None):
    """
    加载指定数据集并标准化格式。
    Args:
        name: 数据集名称或 parquet 文件路径
        base_dir: 数据集根目录，默认为 DEFAULT_BASE_DIR
    Returns:
        dataset: list[dict] 标准化格式的数据集
        dataset_name: 数据集名
    """
    
    
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    # 如果直接传入 .parquet 文件路径
    if name.endswith(".parquet"):
        dataset_path = name
        dataset_name = os.path.splitext(os.path.basename(name))[0]
    else:
        # 构造统一路径: <base_dir>/<dataset_name>.parquet
        dataset_path = os.path.join(base_dir, f"{name}.parquet")
        dataset_name = name
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # 加载原始数据集
    raw_dataset = Dataset.from_parquet(dataset_path)
    
    # 标准化格式
    standardized_dataset = standardize_dataset_format(raw_dataset, dataset_name)
    
    return standardized_dataset, dataset_name


def save_as_dataset(data, save_path):
    """
    保存数据集为 parquet 文件
    Args:
        data: 数据集
        save_path: 保存路径
    """
    newdataset = []
    for item in data:
        for gen in item['responses']:
            newdataset.append({
            'question': item['question'],
            'answer': item['answer'],
            'prompt': item['prompt']+gen
        })
    dataset = Dataset.from_list(newdataset)
    dataset.to_parquet(save_path)