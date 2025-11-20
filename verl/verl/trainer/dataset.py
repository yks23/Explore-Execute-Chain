from datasets import load_dataset,Dataset
import re
def split_answer(answer,type='gsm8k'):
    """
    输入示例 answer 字符串格式：
        "思维链内容 #### 最终答案内容"
    返回：
        cot: 思维链字符串
        final_answer: 最终答案字符串
    """
    if type == 'gsm8k':
        parts = answer.split("####")
        if len(parts) != 2:
            # 不符合预期格式，直接返回原answer和空答案或None
            return answer.strip(), None
        
        cot, final_answer = parts
        return cot.strip(), final_answer.strip()
    elif type == 'math':
        match = re.findall(r"\\boxed\{([^}]*)\}", answer)
        if match:
            return answer,match[-1].strip()  # 通常最后一个就是最终答案
        return answer, None
    elif type == 'modified_gsm8k':
        return answer,answer.split("<answer>")[-1].strip()
    
def get_gsm8k():
# 下载并加载 GSM8K 数据集
    dataset = load_dataset("gsm8k", "main")  # "main" 是标准版本
    return dataset
def get_modified_gsm8k():
    dataset = Dataset.from_parquet('/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/gsm8k_modified/valid.parquet')
    return dataset
def get_math():
    dataset = load_dataset("./dataset/hendrycks_math", "algebra")
    return dataset
def get_aime24():
    # train
    dataset = load_dataset("HuggingFaceH4/aime_2024")
    return dataset
def get_aime25():
    # default,part1,part2
    dataset = load_dataset("yentinglin/aime_2025")
    return dataset
def get_amc23():
    # test
    dataset = load_dataset("math-ai/amc_23")
    return dataset
def get_math500():
    # test
    ds = load_dataset("HuggingFaceH4/MATH-500")
    return ds
def get_minerva():
    # test
    ds = load_dataset("math-ai/minervamath")
    return ds
def get_olympiad_bench():
    # train,
    ds = load_dataset("HuggingFaceH4/olympiad_bench")
    return ds

if __name__ == "__main__":
    name = "openai/gsm8k"
    part = 'main'
    dname= 'math-algebra'
    split_name = 'test'
    path = "/home/fit/alex/Kaisen.Yang/CoT Decomposition/dataset/openrl/test.parquet"
    dataset = Dataset.from_parquet(path)
    final_dataset=[]
    for data in dataset:
        question = data['question']
        solution = data['answer']
        answer = solution.split("<answer>")[-1]  # 获取最后一个部分作为答案
        final_dataset.append({
            'question': question,
            'answer': answer,
            'solution': solution
        })
    print(final_dataset[0])
    final_dataset = Dataset.from_list(final_dataset)
    # import os
    # os.makedirs(f'~/Kaisen.Yang/CoT Decomposition/dataset/{dname}', exist_ok=True)
    final_dataset.to_parquet(path)
    
    
    
    
