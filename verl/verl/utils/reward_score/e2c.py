# Copyright 2025 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
E2C (Explore-Execute-Chain) reward scoring function.

This module provides reward computation for the E2C framework, which evaluates
solutions based on exploration-execution chain alignment and final answer correctness.
"""

import re
from typing import Optional


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


def extract_section_titles(text: str) -> list[str]:
    """
    提取形如 \n1. (something)\n 的小节标题文本内容
    
    Args:
        text: 输入的文本
        
    Returns:
        list: 包含所有小节标题文本的列表
    """
    # 匹配模式：\n数字. (内容)\n，只捕获内容部分
    if not text.endswith('\n'):
        text = text + '\n'
    
    pattern = r'\d+\.\s*(.*?)\n'
    return re.findall(pattern, text)


def lcs_ratio(s1: str, s2: str) -> float:
    """
    Compute the longest common subsequence (LCS) ratio of s1 in s2.
    Optimized for time and space.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        float: Length of LCS divided by len(s1)
    """
    if not s1:
        return 0.0
    s1 = s1.lower().split(' ')
    s2 = s2.lower().split(' ')
    n, m = len(s1), len(s2)
    # Ensure s1 is the shorter one for space optimization
    if n > m:
        s1, s2 = s2, s1
        n, m = m, n

    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_length = prev[m]
    return lcs_length / len(s1)


def correct_format(text: str) -> bool:
    """
    检查文本格式是否正确
    
    Args:
        text: 输入文本
        
    Returns:
        bool: 格式是否正确
    """
    special_segs = ['<EXECUTION>', '</EXECUTION>']
    # for special_seg in special_segs:
    #     if text.count(special_seg) != 1:
    #         return False
    return True


def keep_lowercase_and_digits(s: str) -> str:
    """
    保留字符串中的小写字母和数字，按原顺序拼接。
    
    Args:
        s: 输入字符串
        
    Returns:
        str: 只包含小写字母和数字的字符串
    """
    return ''.join(ch.lower() for ch in s if ch.isascii() and ch.isalnum())


def keep_only_digits(s: str) -> str:
    """
    保留字符串中的数字，按原顺序拼接。
    
    Args:
        s: 输入字符串
        
    Returns:
        str: 只包含数字的字符串
    """
    return ''.join(ch for ch in s if ch.isdigit())


def calculate_frac(s: str) -> Optional[float]:
    """
    此处只考虑单字符串 frac{a}{b}的形式，不然返回None
    
    Args:
        s: 包含分数的字符串
        
    Returns:
        Optional[float]: 分数的浮点值，如果无法解析则返回None
    """
    pattern = r'frac\{(-?\d+)\}\{(-?\d+)\}'
    match = re.search(pattern, s)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator != 0:
            return numerator / denominator
    return None


def normalize_final_answer(final_answer: str) -> tuple[str, Optional[float]]:
    """
    Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        tuple: (normalized_string, optional_float_value)
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
        return final_answer, float_answer
    else:
        return keep_lowercase_and_digits(final_answer), None


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


def get_max_step(text: str) -> int:
    """
    提取文本中的序号步骤，返回最大序号（支持任意位数字）

    Args:
        text: 输入文本

    Returns:
        int: 文本中最大的序号，如果没有找到返回0
    """
    # 匹配任意位数字序号，格式如 1. 或 2)
    matches = re.findall(r'\n(\d)\.\s', text)
    numbers = [int(num) for num in matches]
    return max(numbers) if numbers else 0


def compute_score(
    solution_str: str,
    ground_truth: str,
    prompt_str: str = "",
    use_constrain_reward: bool = False,
    format_score: float = 0.01,
    score: float = 1.0,
    **kwargs
) -> float:
    """
    Compute E2C reward score based on exploration-execution alignment and answer correctness.
    
    Args:
        solution_str: The generated solution string
        ground_truth: The ground truth answer
        prompt_str: The prompt string (optional)
        use_constrain_reward: Whether to use constraint-based reward (default: False)
        format_score: Score penalty for format errors (default: 0.01)
        score: Base score for correct answers (default: 1.0)
        **kwargs: Additional keyword arguments
        
    Returns:
        float: The computed reward score
    """
    if not correct_format(solution_str):
        return 0.0
    
    # Combine prompt and solution
    all_text = kwargs.get('prompt_str', prompt_str) + solution_str
    
    # Extract exploration and execution sections
    if "<EXECUTION>" in all_text:
        exploration_str = all_text.split('<EXECUTION>')[0]
        exploration_str = exploration_str.split('<EXPLORATION>')[-1]
        execution_str = all_text.split('<EXECUTION>')[-1]
        exp_titles = extract_section_titles(exploration_str.replace('*', ''))
        exe_titles = extract_section_titles(execution_str.replace('*', ''))
    elif "</EXPLORATION>" in all_text:
        exploration_str = all_text.split('</EXPLORATION>')[0]
        exploration_str = exploration_str.split('<EXPLORATION>')[-1]
        execution_str = all_text.split('</EXPLORATION>')[-1]
        exp_titles = extract_section_titles(exploration_str.replace('*', ''))
        exe_titles = extract_section_titles(execution_str.replace('*', ''))
    else:
        # 用\n1. 作为分割
        exp_titles = all_text.split('\n1. ')[-1]
        exp_titles = extract_section_titles(exp_titles.replace('*', ''))
        exe_titles = ("".join(all_text.split('\n1. ')[:-1])).split('<EXECUTION>')[-1]
        exe_titles = extract_section_titles(exe_titles.replace('*', ''))
    
    # Calculate alignment score between exploration and execution
    cnt_have = 0
    for title1 in exp_titles:
        for title2 in exe_titles:
            if title1 in title2 or lcs_ratio(title1, title2) > 0.3:
                cnt_have += 1
                break
    
    base_score = cnt_have / max(len(exp_titles), 1)
    
    # Normalize ground truth
    gt, p_float = normalize_final_answer(ground_truth)
    succ = False
    
    # Apply constraint reward if enabled
    if not use_constrain_reward:
        base_score = 1
    
    # Extract predicted answer from the last 300 characters
    if len(extract_boxed_content(solution_str[-300:])) == 0:
        return 0
    
    pre_answer = extract_boxed_content(solution_str[-300:])[-1]
    pre_answer, q_float = normalize_final_answer(pre_answer)
    
    # Check if answer is correct
    if pre_answer == gt or (q_float is not None and q_float == p_float):
        succ = True
    
    if succ:
        return base_score
    else:
        return 0

