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
def load_dataset_by_name(name: str, base_dir: str = None):
    """
    加载指定数据集。
    Args:
        name: 数据集名称或 parquet 文件路径
        base_dir: 数据集根目录，默认为 DEFAULT_BASE_DIR
    Returns:
        dataset: HuggingFace Dataset
        dataset_name: 数据集名
    """
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR

    # 如果直接传入 .parquet 文件路径
    if name.endswith(".parquet"):
        return Dataset.from_parquet(name), os.path.splitext(os.path.basename(name))[0]

    # 构造统一路径: <base_dir>/<dataset_name>/<dataset_name>.parquet
    dataset_path = os.path.join(base_dir, name, f"{name}.parquet")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    return Dataset.from_parquet(dataset_path), name
