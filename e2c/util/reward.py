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

import re

def shift_numbered_list(text: str, k: int) -> str:
    """
    将文中形如 '1. ', '2. ' 的序号整体加上 k
    
    参数:
        text (str): 输入文本
        k (int): 偏移量
        
    返回:
        str: 修改后的文本
    """
    def replacer(match):
        num = int(match.group(1))
        return f"{num + k}. "
    
    return re.sub(r'\b(\d+)\.\s', replacer, text)

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

# s  = "Some text \\boxed{\\frac{4}{3}} and \\boxed{second} end."
# print(extract_boxed_content(s))