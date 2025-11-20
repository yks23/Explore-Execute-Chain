import torch

def compute_success_ratio(generated_texts,answers):
    results = []
    
    for batch_sample,answer in zip(generated_texts,answers):
        batch_success = []
        for text in batch_sample:
            # 归一化
            boxed_answer = extract_boxed_content(text)
            if len(boxed_answer)==0:
                continue
            else:
                for boxed in boxed_answer:
                    if boxed == answer:
                        batch_success.append(1)
                        break
            
        results.append(
         {
             "all_samples": batch_sample,
             "success": batch_success,
             "avg_success": sum(batch_success)/len(batch_success) if batch_success else 0,
             "best_success": 1 if any(batch_success) else 0
         }   
        )
    return results

def generate_and_compute_entropy(
    model,
    tokenizer,
    input_text,
    answers,
    max_new_tokens,
    device="cuda",
    sample_num=1,
    temperature=1.0,
    top_p=1.0
):
    """支持多次采样（高效版，一次前向生成多样本）"""
    # 批处理输入
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        padding_side="left"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=sample_num,
            return_dict_in_generate=True,
            output_scores=False,
          
        )
    generated_ids = outputs.sequences  # (batch_size * sample_num, seq_len)
    # 解码
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
    # 分组成[B,Sample,L]
    batch_size = input_ids.input_ids.shape[0]
    grouped_texts = [
        generated_texts[i * sample_num:(i + 1) * sample_num] for i in range(batch_size)
    ]
    
    results = compute_success_ratio(grouped_texts,answers)
    return results
    
def get_all_success(val_dataset,model,tokenizer,cfg):
    """
    val_dataset: 并行dataset
    model: 并行model
    tokenizer: tokenizer
    cfg: 配置
    
    本函数要传入并行的dataset、模型，完成计算之后把不同进程的收集了，然后返回
    """
    answer_key = cfg.get("answer_key","answer")
    question_key = cfg.get("question_key","problem")
    answers = [data[answer_key] for data in val_dataset]
    questions = [data[question_key] for data in val_dataset]
    
    all_success = []
    batch_size = cfg.get("eval_batch_size",16)
    max_new_tokens = cfg.get("max_new_tokens",256)
    sample_num = cfg.get("sample_num",16)
    temperature = cfg.get("temperature",1.0)
    top_p = cfg.get("top_p",1.0)
    device = cfg.get("device","cuda")
    for i in range(0,len(questions),batch_size):
        input_texts = questions[i:i+batch_size]
        batch_answers = answers[i:i+batch_size]
        batch_success = generate_and_compute_entropy(
            model=model.module,
            tokenizer=tokenizer,
            input_text=input_texts,
            answers=batch_answers,
            max_new_tokens=max_new_tokens,
            device=device,
            sample_num=sample_num,
            temperature=temperature,
            top_p=top_p
        )
        all_success.extend(batch_success)
    return all_success
    
    

def extract_boxed_content(s: str) -> list[str]:
    """
    Parses a string and extracts all content enclosed by \\boxed{}.

    Args:
        s: The input string.

    Returns:
        A list of strings, where each string is the content found inside a \\boxed{} block.
    """
    results = []
    # Use a counter to track the nesting level of curly braces
    brace_level = 0
    # Store the starting index of a \\boxed{} block
    start_index = -1
    
    i = 0
    while i < len(s):
        # Look for the start of a new \\boxed{} block
        if s[i:i+6] == "boxed{":
            # If we are not already inside a block, set the start index
            if brace_level == 0:
                start_index = i + 6
            # Increment the brace level
            brace_level += 1
            # Move the index past the opening part of \\boxed{}
            i += 6
            continue
        
        # If we are inside a \\boxed{} block (brace_level > 0)
        if brace_level > 0:
            if s[i] == "{":
                brace_level += 1
            elif s[i] == "}":
                brace_level -= 1
                # If brace_level drops to 0, we have found a complete \\boxed{} block
                if brace_level == 0:
                    content = s[start_index:i]
                    results.append(content)
                    start_index = -1  # Reset start index
        
        i += 1
            
    return results

def multi_choice_evaluate(pred, gt):
    
    if 'boxed{' in pred:
        boxed_answer = pred.split("boxed{")[-1]
        boxed_answer = boxed_answer[:10]
        boxed_answer = boxed_answer.split("}")[0]
        if len(boxed_answer)==1:
            if boxed_answer.isdigit():
                boxed_answer = {"1":"A","2":"B","3":"C","4":"D","5":"E"}.get(boxed_answer,boxed_answer)
            if boxed_answer == gt['answer_idx']:
                return True
            else:
                return False
    else:
        # last_words = pred.strip().split('\n')[-1].split(' ')[-10:]
        # answer_words = gt['answer'].split(' ')
        
        # for ans in answer_words:
        #     find = False
        #     for word in last_words:
        #         if lcs_ratio(ans, word) > 0.9:
        #             find = True
        #             break
        #     if not find:
        #         return False
        # return True
        return None
    return False
    
    
    
def boxed_evaluate(pred, gt):
    gt,p_float = normalize_final_answer(gt)
    pre_answer = ""
    # find every boxed and compare with gt
    for pre_answer in extract_boxed_content(pred):
        pre_answer, q_float = normalize_final_answer(pre_answer)
        if pre_answer == gt or (q_float is not None and q_float == p_float):
            return True , pre_answer
    return False , pre_answer



def normalize_final_answer(final_answer: str) -> str:
        """Normalize a final answer to a quantitative reasoning question.

        Args:
            final_answer: The answer string to normalize

        Returns:
            Normalized answer string
        """
        final_answer = final_answer.split("=")[-1]

        # Apply substitutions and removals
        for before, after in SUBSTITUTIONS:
            final_answer = final_answer.replace(before, after)
        for expr in REMOVED_EXPRESSIONS:
            final_answer = final_answer.replace(expr, "")

        # Extract and normalize LaTeX math
        final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
        final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
        final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

        # Normalize shorthand TeX:
        #  \fracab -> \frac{a}{b}
        #  \frac{abc}{bef} -> \frac{abc}{bef}
        #  \fracabc -> \frac{a}{b}c
        #  \sqrta -> \sqrt{a}
        #  \sqrtab -> sqrt{a}b
        final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
        final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
        final_answer = final_answer.replace("$", "")

        # Normalize numbers
        if final_answer.replace(",", "").isdigit():
            final_answer = final_answer.replace(",", "")
        # normalize frac
        if 'frac' in final_answer:
            # 把保留数字就行
            float_answer = calculate_frac(final_answer)
            final_answer = keep_only_digits(final_answer)
            return final_answer,float_answer
        else:
            return keep_lowercase_and_digits(final_answer),None
        
        
import re
# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]
def keep_lowercase_and_digits(s: str) -> str:
    """
    保留字符串中的小写字母和数字，按原顺序拼接。
    """
    return ''.join(ch.lower() for ch in s if ch.isascii() and ch.isalnum())

def keep_only_digits(s: str) -> str:
    """
    保留字符串中的数字，按原顺序拼接。
    """
    return ''.join(ch for ch in s if ch.isdigit())

def calculate_frac(s: str) -> float:
    """
    此处只考虑单字符串 frac{a}{b}的形式，不然返回none
    """
    pattern = r'frac\{(-?\d+)\}\{(-?\d+)\}'
    match = re.search(pattern, s)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator != 0:
            return numerator / denominator
    return None