import random
from typing import List, Tuple, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import embed_plans, cluster_plans
from ..inference import tts_prompts.LLM_COMBINATION_PROMPT


def _lcs_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.lower(), s2.lower()
    n, m = len(s1), len(s2)
    # 空间复杂度 O(min(n, m))
    if n < m:
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
    return lcs_length / max(len(s1), len(s2))

def _find_best_matching_plan_lcs(generated_text: str, original_plans: List[str]) -> str:
    if not original_plans:
        return ""
    generated_text = generated_text.strip()
    scores = [_lcs_ratio(generated_text, plan) for plan in original_plans]
    best_match_index = scores.index(max(scores))
    return original_plans[best_match_index]

def e2c_select_self_lm_judge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    plans: List[str],
    question: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> Tuple[List[str], List[float]]:

    if not plans:
        return [], []
    candidate_plans_str = ""
    for i, p in enumerate(plans):
        clean_plan = p.replace("<exploration>", "").replace("</exploration>", "").strip()
        candidate_plans_str += f"Plan {i + 1}: <exploration>{clean_plan}</exploration>\n"
    prompt = LLM_COMBINATION_PROMPT.format(
        problem=question,
        explorations=candidate_plans_str.strip()
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    selected_plan_text = tokenizer.decode(outputs[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True)
    best_match_plan = _find_best_matching_plan_lcs(selected_plan_text, plans)
    return [best_match_plan], [1.0]
def e2c_select_semantic_cluster(
        plans: List[str],
        num_clusters: int,
        embedding_model_name: str,
) -> Tuple[List[str], List[float]]:
    if not plans:
        return [], []

    plan_embeddings = embed_plans(plans, embedding_model_name)
    centroid_indices, cluster_sizes = cluster_plans(plan_embeddings, num_clusters)
    selected_plans = [plans[i] for i in centroid_indices]
    weights = [float(size) for size in cluster_sizes]
    return selected_plans, weights

def e2c_sc(
        plans: List[str]
) -> Tuple[List[str], List[float]]:
    num_plans = len(plans)
    weights = [1.0] * num_plans
    return plans, weights
def e2c_rp(
        plans: List[str]
) -> Tuple[List[str], List[float]]:
    if not plans:
        return [], []
    selected_plan = random.choice(plans)
    return [selected_plan], [1.0]

STRATEGY_DISPATCHER = {
    "self_lm_judge": e2c_select_self_lm_judge,
    "semantic_cluster": e2c_select_semantic_cluster,
    "self_consistency": e2c_sc,
    "random_plan": e2c_rp,
}
def apply_tts_strategy(
        strategy_name: str,
        plans: List[str],
        **kwargs
) -> Tuple[List[str], List[float]]:

    if strategy_name not in STRATEGY_DISPATCHER:
        raise ValueError(
            f"未知的 TTS 策略: '{strategy_name}'. "
            f"可用策略包括: {list(STRATEGY_DISPATCHER.keys())}"
        )
    strategy_func = STRATEGY_DISPATCHER[strategy_name]
    import inspect
    sig = inspect.signature(strategy_func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return strategy_func(plans=plans, **valid_kwargs)