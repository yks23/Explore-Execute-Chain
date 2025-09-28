import os
import json
import hydra
import torch
from typing import List
import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from util.reward import normalize_final_answer,extract_boxed_content,shift_numbered_list
from util.visualization import plot_by_task_samples
from util.dataset import max_token_dataset, max_batch_size, is_multi_choice, load_dataset_by_name
from util.model import EntropyCalculator,check_resume, load_model,TemperatureProcessor
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def generate_and_compute_entropy(
    model,
    tokenizer,
    input_text,
    max_new_tokens,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=0.7,
    need_static=True
):
    # 批处理输入
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        padding_side="left"
    ).to(device)
    B = len(input_text)  # batch size
    model.eval()
    model.to(device)
    logit_processors = []
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=sample_num,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=[tokenizer.eos_token_id] if stop_token is None else [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(stop_token)[0],
            ],
            logits_processor=logit_processors
        )
        logits = outputs.scores  # list[step](B*sample_num, vocab)
        sequences = outputs.sequences[:, input_ids.input_ids.shape[1]:]  # only new tokens
        total_size = B * sample_num
        assert sequences.shape[0] == total_size

        # 初始化按 sample 排列
        all_results = []

        for s in range(sample_num):
            # 每个 batch 的第 s 个 sample
            batch_tokens = sequences[s::sample_num]  # (B, L)
            
            # 计算 entropy: steps x B -> B x L
            step_entropies = []
            step_confidences = []
            for step, logit in enumerate(logits):
                step_logit = logit[s::sample_num]  # (B, vocab)
                step_prob = torch.softmax(step_logit, dim=-1)
                step_entropy = -torch.sum(step_prob * torch.log(step_prob + 1e-8), dim=-1)
                
                step_confidence = torch.topk(step_prob, k=5, dim=-1).values
                step_confidence = torch.mean(torch.log(step_confidence + 1e-8), dim=-1)
                
                step_confidences.append(step_confidence.cpu())  # (B,)
                step_entropies.append(step_entropy.cpu())  # (B,)

            # 转置 step_entropies -> B x L
            batch_entropy = torch.stack(step_entropies, dim=1).tolist()  # (B, L)
            batch_confidence = torch.stack(step_confidences, dim=1).tolist()  # (B, L)
            # 每个 sample 对应 batch 内每个输入的 (tokens, entropy)
            tokens= []
            entropies=[]
            confidences =[]
            for i in range(B):
                tokens.append(batch_tokens[i].cpu().tolist())
                entropies.append(batch_entropy[i])
                confidences.append(batch_confidence[i])
                

            all_results.append((tokens, entropies, confidences))
    if need_static:
        return all_results
    else:
        return [all_results[i][0] for i in range(len(all_results))]

def calculate_entropy(token_logprob_dict):
    entropy = 0.0
    for id,logprob in token_logprob_dict.items():
        prob = torch.exp(torch.tensor(logprob.logprob))
        entropy -= prob * logprob.logprob
    return entropy.item()

def calculate_confidence(token_logprob_dict,K=5):
    sorted_logprobs = sorted([logprob.logprob for id, logprob in token_logprob_dict.items()], reverse=True)
    top_k_logprobs = sorted_logprobs[:K]
    confidence = -sum(top_k_logprobs)
    return confidence

def generate_and_compute_entropy_vllm(
    model: LLM,
    input_text: List[str],
    max_new_tokens: int,
    sample_num: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.7,
    stop_token: str = None
):
    """
    使用 vllm 生成文本并获取每个生成 token 的对数概率。

    Args:
        model (LLM): vLLM 实例。
        input_text (List[str]): 输入提示的列表。
        max_new_tokens (int): 生成的新 token 的最大数量。
        sample_num (int): 每个提示生成的样本数量。
        temperature (float): 采样温度。
        top_p (float): top-p 采样值。
        stop_token (str): 停止生成的 token。

    Returns:
        一个元组列表。每个元组包含一个生成的文本列表和每个文本对应的 token 对数概率列表。
    """
    
    # 配置采样参数
    sampling_params = SamplingParams(
        n=sample_num,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop_token_ids=[model.get_tokenizer().convert_tokens_to_ids(stop_token)] if stop_token else None,
        logprobs=1  # 请求每一步生成的 token 的对数概率
    )

    # 生成文本和对数概率
    outputs = model.generate(input_text, sampling_params)

    results = []
    
    for output in outputs:
        generated_texts = []
        token_logprobs = []
        confidence_list = []
        # 获取批次内每个样本的结果
        for sequence in output.outputs:
            generated_texts.append(sequence.text)
            
            # logprobs 对象是字典的列表，每个字典对应一个 token
            logprob_list = [calculate_entropy(token) for token in sequence.logprobs]
            
            confidence_list.append([calculate_confidence(token) for token in sequence.logprobs])
            
            token_logprobs.append(logprob_list)

        results.append((generated_texts, token_logprobs, confidence_list))

    return results



def decode_predictions(tokenizer, tokens, special_tokens, answer_seg):
    pred_texts = [
        decode_with_selected_special_tokens(tokenizer, special_tokens, ids) for ids in tokens
    ]
    answers = [txt[-200:] for txt in pred_texts]
    return pred_texts, answers


def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
            """
            解码 token 序列，仅保留指定的特殊 token，移除其他特殊 token。

            参数:
            - tokenizer: Tokenizer 实例，用于解码和转换 token。
            - special_tokens: 需要保留的特殊 token 的字符串列表。
            - token_ids: 待解码的 token 序列。

            返回:
            - 过滤后的解码文本字符串。
            """
            # 获取保留的特殊 token 的 ID
            special_tokens +=['<THINKING>','</THINKING>','<EXPLORATION>','</EXPLORATION>','<EXECUTION>','</EXECUTION>']
            keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
            
            # 解码之前，过滤掉不在保留列表中的特殊 token
            filtered_token_ids = [
                token_id for token_id in token_ids
                if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
            ]
            
            # 使用过滤后的 token_ids 进行解码
            decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
            return decoded_text
def check_format(text, checklist):
    return all(key in text for key in checklist)

def lcs_ratio(s1: str, s2: str) -> float:
    """
    Compute the longest common subsequence (LCS) ratio of s1 in s2.
    Optimized for time and space.
    Returns: length of LCS divided by len(s1)
    """
    if not s1:
        return 0.0
    s1 = s1.lower()
    s2 = s2.lower()
    n, m = len(s1), len(s2)
    # Ensure s1 is the shorter one for space optimization
    # if n > m:
        # s1, s2 = s2, s1
        # n, m = m, n

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

def multi_choice_evaluate(pred, gt):
    
    if 'boxed{' in pred:
        boxed_answer = pred.split("boxed{")[-1]
        boxed_answer = boxed_answer.split("text{")[-1]
        boxed_answer = boxed_answer.split("}")[0]
        if len(boxed_answer)==1:
            if boxed_answer.isdigit():
                boxed_answer = {"1":"A","2":"B","3":"C","4":"D","5":"E"}.get(boxed_answer,boxed_answer)
            if boxed_answer == gt['answer_idx']:
                return True,boxed_answer
            else:
                return False,boxed_answer
    else:
        last_choice = ''
        for char in reversed(pred):
            if char in ['A','B','C','D','E']:
                last_choice = char
                break
            
        if last_choice:
            if last_choice == gt['answer_idx']:
                return True,last_choice
            else:
                return False,last_choice
        else:
            return None,"None"
        
    return False,"None"
    
    
    
def boxed_evaluate(pred, gt):
    gt,p_float = normalize_final_answer(gt)
    pre_answer = ""
    # find every boxed and compare with gt
    for pre_answer in extract_boxed_content(pred):
        pre_answer, q_float = normalize_final_answer(pre_answer)
        if pre_answer == gt or (q_float is not None and q_float == p_float):
            return True , pre_answer
    return False , pre_answer


def evaluate_batch(questions, gt_answers, samples, cfg, tokenizer):
    K = len(questions)
    sample_num = len(samples)
    entropy_calculator = EntropyCalculator(tokenizer)
    all_pred_texts = []
    all_answers = []
    for tokens, _,_ in samples:
        pred_texts, answers = decode_predictions(tokenizer, tokens, cfg.eval.head + cfg.eval.tail, cfg.eval.answer_seg)
        all_pred_texts.append(pred_texts)
        all_answers.append(answers)
    entropy_stats = {}
    if samples:
        all_tokens = []
        all_entropies=[]
        all_confidences = []
        for tokens, entropies,confidences in samples:
            all_tokens.extend(tokens)
            all_entropies.extend(entropies)
            all_confidences.extend(confidences)
        for start_token, end_token in zip(cfg.eval.head, cfg.eval.tail):
            stats = entropy_calculator.calculate_batch_entropy_stats(
                (all_tokens, all_entropies,all_confidences),start_token, end_token
            )
            key = f"{start_token}_{end_token}"
            entropy_stats[key] = stats
    results = []
    success_avg, success_best, format_avg = 0, 0, 0

    for i in range(K):
        per_sample_success = []
        per_sample_format = []
        per_sample_outputs = []
        failed_samples = []
        for s in range(sample_num):
            pred = all_answers[s][i]
            
            tokens = samples[s][0][i]
            entropies = samples[s][1][i]
            confidences = samples[s][2][i]
            full_text = all_pred_texts[s][i]
            boxed_count = full_text.count("boxed")
            
            if isinstance(gt_answers[i],dict):
                succ,pre_answer = multi_choice_evaluate(full_text, gt_answers[i])
                if succ==None:
                    failed_samples.append(full_text)
                    continue
            else:
                succ,pre_answer = boxed_evaluate(full_text, gt_answers[i])

            fmt = check_format(full_text, cfg.eval.checklist)
            per_sample_success.append(succ)
            per_sample_format.append(fmt)
            
            per_sample_outputs.append({
                "answer": pre_answer,
                "full_text": full_text,
                "success": succ,
                "format_success": fmt,
                "static": entropy_calculator.calculate_entropy_stats((tokens, entropies,confidences),cfg.eval.head[0], cfg.eval.tail[0]) if cfg.eval.need_static else None,
                "boxed_count": boxed_count
            })

        # 平均成功率：样本成功率的均值
        success_avg += sum(per_sample_success) / len(per_sample_success) if per_sample_success else 0.25
        format_avg += sum(per_sample_format) / len(per_sample_format) if per_sample_format else 0.25
        # 最好成功率：只要有一次成功就算成功
        success_best += 1 if any(per_sample_success) else 0.25

        results.append({
            "question": questions[i],
            "gt_answer": gt_answers[i],
            "samples": per_sample_outputs,
            "failed_samples": failed_samples,
            "avg_success": sum(per_sample_success) / len(per_sample_success) if per_sample_success else 0.25,
            "best_success": 1 if any(per_sample_success) else 0,
            "finish_rate": sum([1 for out in per_sample_outputs if out['boxed_count']>0]) / len(per_sample_outputs) if per_sample_outputs else 0, 
            "finish_and_correct_rate": sum([1 for out in per_sample_outputs if out['boxed_count']>0 and out['success']]) / len(per_sample_outputs) if per_sample_outputs else 0,
        })

    batch_stats = {
        "avg_success": success_avg / K,
        "best_success": success_best / K,
        "format_avg": format_avg / K,
        "all_number": K,
        "entropy_stats": entropy_stats  # 添加熵统计
    }
    
    return results, batch_stats
def log_batch_stats(batch_idx, batch_stats, rank):
    if rank != 0:
        return
        
    avg_rate = batch_stats["avg_success"]
    best_rate = batch_stats["best_success"]
    fmt_rate = batch_stats["format_avg"]
    
    print(f"\n[Batch {batch_idx}] 🎲 平均成功率: {avg_rate:.2%} | 最好成功率: {best_rate:.2%} | 格式率: {fmt_rate:.2%}")
    if "entropy_stats" in batch_stats:
        print("📊 熵统计:")
        for key, stats in batch_stats["entropy_stats"].items():
            if stats["sample_count"] > 0:
                print(f"   {key}: 长度={stats['avg_length']:.1f}, "
                      f"总熵={stats['avg_total_entropy']:.2f}, "
                      f"平均熵={stats['avg_entropy_per_token']:.4f} "
                      f"({stats['sample_count']}样本)")

def get_distributed_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    
    # 创建数据加载器
    indices = list(sampler)
    return [dataset[i] for i in indices], len(indices)
def merge_results_from_all_ranks(save_path, seed, world_size,save_head=False):
    if torch.distributed.get_rank() != 0:
        return
        
    all_results = []
    all_static = {"avg_success": 0, "best_success": 0, "format_avg": 0, "all_number": 0, "entropy_stats": {}}
    total_samples = 0
    finish_count = 0
    finish_and_correct_count = 0
    for rank in range(world_size):
        result_file = os.path.join(save_path, f"result_{seed}_rank{rank}.json")
        static_file = os.path.join(save_path, f"static_{seed}_rank{rank}.json")
        
        if os.path.exists(result_file) and os.path.exists(static_file):
            with open(result_file, "r") as f:
                results = json.load(f)
            with open(static_file, "r") as f:
                static = json.load(f)
            
            all_results.extend(results)
            
            # 合并统计信息
            if static["all_number"] > 0:
                weight = static["all_number"]
                all_static["avg_success"] = (all_static["avg_success"] * total_samples + static["avg_success"] * weight) / (total_samples + weight)
                all_static["best_success"] = (all_static["best_success"] * total_samples + static["best_success"] * weight) / (total_samples + weight)
                all_static["format_avg"] = (all_static["format_avg"] * total_samples + static["format_avg"] * weight) / (total_samples + weight)
                total_samples += weight
                
                # 合并熵统计
                for key, stats in static.get("entropy_stats", {}).items():
                    if key not in all_static["entropy_stats"]:
                        all_static["entropy_stats"][key] = stats.copy()
                    else:
                        old_stats = all_static["entropy_stats"][key]
                        total_sample_count = old_stats["sample_count"] + stats["sample_count"]
                        
                        if total_sample_count > 0:
                            all_static["entropy_stats"][key] = {
                                "avg_length": (old_stats["avg_length"] * old_stats["sample_count"] + stats["avg_length"] * stats["sample_count"]) / total_sample_count,
                                "avg_total_entropy": (old_stats["avg_total_entropy"] * old_stats["sample_count"] + stats["avg_total_entropy"] * stats["sample_count"]) / total_sample_count,
                                "avg_entropy_per_token": (old_stats["avg_entropy_per_token"] * old_stats["sample_count"] + stats["avg_entropy_per_token"] * stats["sample_count"]) / total_sample_count,
                                "sample_count": total_sample_count
                            }
    entropies=[]
    for result in all_results:
        avg_success = result.get("avg_success", 0)
        success_entropy = -avg_success*torch.log(torch.tensor(avg_success) + 1e-8) - (1 - avg_success)*torch.log(torch.tensor(1 - avg_success) + 1e-8)
        entropies.append(success_entropy.item())
        finish_count += sum([1 for out in result['samples'] if out['boxed_count']>0])
        finish_and_correct_count += sum([1 for out in result['samples'] if out['boxed_count']>0 and out['success']])
    all_static['finish_rate'] = finish_count/ total_samples if total_samples>0 else 0
    all_static['finish_and_correct_rate'] = finish_and_correct_count/ finish_count if finish_count>0 else 0
    all_static['success_entropy'] = sum(entropies)/len(entropies) if len(entropies) > 0 else 0
    
    if save_head:
        q2a = {}
        for result in all_results:
            question = result['question']
            answer = result['samples'][0]['full_text']
            q2a[question] = answer
        with open(os.path.join(save_path, f"q2a.json"), "w") as f:
            json.dump(q2a, f, ensure_ascii=False, indent=4)
    
    # 保存合并后的结果
    if all_results:
        with open(os.path.join(save_path, f"result_{seed}_merged.json"), "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(save_path, f"static_{seed}_merged.json"), "w") as f:
            json.dump(all_static, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 合并完成: 总共 {len(all_results)} 个样本")
        print(f"📊 最终统计: 平均成功率={all_static['avg_success']:.2%}, "
              f"最好成功率={all_static['best_success']:.2%}, "
              f"格式率={all_static['format_avg']:.2%}")
    
    
    
    plot_by_task_samples(all_results, save_path,10)
import re
def get_max_step(text: str) -> int:
    matches = re.findall(r'\b(\d+)[\.\)]', text)
    numbers = [int(num) for num in matches]
    return max(numbers) if numbers else 0

def merge_exploration_part(explorations:list[str]):
    start_str = "I need to make a step-by-step plan."
    start_str +=explorations[0]
    for i in range(1,len(explorations)):
        e = explorations[i].split('</EXPLORATION>')[0].replace('<EXECUTION>','').replace('</EXECUTION>','').strip()
        e = '\n1. '+"I will try another method "+ e.split('\n1. ')[-1]
        steps = get_max_step(start_str)
        start_str += shift_numbered_list(e,steps)
    all_steps = get_max_step(start_str)
    if len(explorations)>1:
        start_str += f" {all_steps+1}. " + "Combine the above methods and give a final answer in boxed.</EXPLORATION>"
    else:
        start_str += "</EXPLORATION>"
    return start_str

@hydra.main(version_base=None, config_path="config", config_name="eval")
def eval_main(cfg):
    ddp_setup()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = f"cuda:{rank}"
    torch.manual_seed(cfg.eval.seed + rank)  # 不同rank使用不同的随机种子
    model, tokenizer = load_model(cfg.model, device)
        
        # 使用DDP包装模型
    model = DDP(model, device_ids=[rank])
    
    overall_stats = {}
    
    if 'all' in cfg.eval.dataset:
        cfg.eval.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    if 'med' in cfg.eval.dataset:
        cfg.eval.dataset = ["clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","anatomy","medqa","medmcqa"]
    for dataset_name in cfg.eval.dataset:
        dataset, dataset_name = load_dataset_by_name(dataset_name)
        if cfg.eval.get('use_default',False):
            if dataset_name in is_multi_choice:
                cfg.eval.system_prompt = r"You are a medical expert. You will be given a medical question and several candidate answers. Please choose the best answer based on your medical knowledge. You must give your answer in the format: 'The correct answer is boxed{A,B,C or D}'."
                cfg.eval.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."
            else:
                cfg.eval.system_prompt = r""
                cfg.eval.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."
        
        if cfg.eval.batch_size == -1:
            batch_size = max_batch_size.get(dataset_name, 10) // cfg.eval.sample_num  # 每个rank的batch size
        else:
            batch_size = cfg.eval.batch_size // cfg.eval.sample_num  # 每个rank的batch size
        if cfg.eval.max_new_tokens == -1:
            max_tokens = max_token_dataset.get(dataset_name, 1000)
        else:
            max_tokens = cfg.eval.max_new_tokens    
        
        save_path = os.path.join(cfg.eval.save_path, dataset_name)
        
        if cfg.eval.get("combine_num",0)>1:
            with open(os.path.join(save_path,'result_0_merged.json'),'r') as f:
                old_results = json.load(f)
            q2hint = {}
            for res in old_results:
                explorations = [res['samples'][i]['full_text'] for i in range(len(res['samples']))][:cfg.eval.combine_num]
                explorations = [e.replace('<EXPLORATION>','').replace('</EXPLORATION>','') for e in explorations]
                hint = merge_exploration_part(explorations)
                q2hint[res['question']] = hint
                    
                    
        
        os.makedirs(save_path, exist_ok=True)
        if cfg.eval.get('resume', False):
            results, static = check_resume(save_path, cfg.eval.seed, rank)
            print(f"[Rank {rank}] Resuming evaluation for dataset {dataset_name} from {len(results)} existing results.")
        else:
            results, static = [], {}
        
        subset, subset_size = get_distributed_dataloader(dataset, batch_size, rank, world_size)
        if rank == 0:
            bar = tqdm.tqdm(total=len(dataset), desc=f"Processing {dataset_name}", position=0, leave=True)
        else:
            bar = None
        if static and "all_number" in static and static["all_number"] > 0:
            local_cum_processed = int(static["all_number"])
            local_avg_succ_sum = float(static.get("avg_success", 0.0)) * local_cum_processed
            local_best_succ_sum = float(static.get("best_success", 0.0)) * local_cum_processed
            local_fmt_sum = float(static.get("format_avg", 0.0)) * local_cum_processed
        else:
            local_cum_processed = 0
            local_avg_succ_sum = 0.0
            local_best_succ_sum = 0.0
            local_fmt_sum = 0.0
        init_buf = torch.tensor(
            [local_cum_processed, local_avg_succ_sum, local_best_succ_sum, local_fmt_sum],
            device=device, dtype=torch.float64
        )
        if cfg.eval.backend=='hf':
            torch.distributed.all_reduce(init_buf, op=torch.distributed.ReduceOp.SUM)
        if rank == 0:
            global_processed_init = int(init_buf[0].item())
            if global_processed_init > 0:
                bar.n = global_processed_init
                bar.refresh()
                avg_success_init = init_buf[1].item() / global_processed_init
                best_success_init = init_buf[2].item() / global_processed_init
                fmt_init = init_buf[3].item() / global_processed_init
                print(f"[Init Global] processed={global_processed_init} | "
                      f"avg_success={avg_success_init:.2%} | best_success={best_success_init:.2%} | format={fmt_init:.2%}")
        start_idx = len(results)
        for i in range(start_idx, subset_size, batch_size):
            batch_data = subset[i:i + batch_size]
            guidelines = [d.get('exploration', '') for d in batch_data]
            if cfg.eval.get("system_prompt",None)is not None:
                system_prompt = [{
                    "role": "system",
                    "content": cfg.eval.system_prompt
                }]
                print(system_prompt)
            else:
                system_prompt = []
            questions = [
                tokenizer.apply_chat_template(
                    system_prompt+[
                        {"role": "user", "content": cfg.eval.get('question_prefix','')+q["question"] + cfg.eval.get('question_suffix','').replace("<guideline>", g)}
                        ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=cfg.eval.get("enable_thinking", True)
                ) + cfg.eval.get("solution_prefix", "").replace("<guideline>", g)
                for q, g in zip(batch_data, guidelines)
            ]
            
            if cfg.eval.get("combine_num",0)>1:
                questions = [q+q2hint[q] for q in questions]
                
            gt_answers = [d["answer"] for d in batch_data]
            samples = generate_and_compute_entropy(
                model.module, tokenizer, questions,
                max_tokens,
                device=device,
                stop_token=cfg.eval.get("stop_token", None),
                sample_num=cfg.eval.sample_num,
                temperature=cfg.eval.temperature,
                top_p=cfg.eval.top_p,
            )

            batch_results, batch_stats = evaluate_batch(questions, gt_answers, samples, cfg, tokenizer)
            results.extend(batch_results)
            if not static:
                static = {"avg_success": 0, "best_success": 0, "format_avg": 0, "all_number": 0, "entropy_stats": {}}
            static["avg_success"] = (static["avg_success"] * static["all_number"] + batch_stats["avg_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["best_success"] = (static["best_success"] * static["all_number"] + batch_stats["best_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["format_avg"] = (static["format_avg"] * static["all_number"] + batch_stats["format_avg"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
            static["all_number"] += batch_stats["all_number"]
            if "entropy_stats" in batch_stats:
                for key, new_stats in batch_stats["entropy_stats"].items():
                    if key not in static["entropy_stats"]:
                        static["entropy_stats"][key] = new_stats.copy()
                    else:
                        old_stats = static["entropy_stats"][key]
                        total_samples = old_stats["sample_count"] + new_stats["sample_count"]
                        if total_samples > 0:
                            static["entropy_stats"][key] = {
                                "avg_length": (old_stats["avg_length"] * old_stats["sample_count"] + new_stats["avg_length"] * new_stats["sample_count"]) / total_samples,
                                "avg_total_entropy": (old_stats["avg_total_entropy"] * old_stats["sample_count"] + new_stats["avg_total_entropy"] * new_stats["sample_count"]) / total_samples,
                                "avg_entropy_per_token": (old_stats["avg_entropy_per_token"] * old_stats["sample_count"] + new_stats["avg_entropy_per_token"] * new_stats["sample_count"]) / total_samples,
                                "sample_count": total_samples
                            }
            with open(os.path.join(save_path, f"result_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            with open(os.path.join(save_path, f"static_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
                json.dump(static, f, ensure_ascii=False, indent=4)
            local_cum_processed = int(static["all_number"])
            local_avg_succ_sum += batch_stats["avg_success"] * batch_stats["all_number"]
            local_best_succ_sum += batch_stats["best_success"] * batch_stats["all_number"]
            local_fmt_sum += batch_stats["format_avg"] * batch_stats["all_number"]

            buf = torch.tensor(
                [local_cum_processed, local_avg_succ_sum, local_best_succ_sum, local_fmt_sum],
                device=device, dtype=torch.float64
            )
            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)

            if rank == 0:
                global_processed = int(buf[0].item())
                if global_processed > 0:
                    avg_success = buf[1].item() / global_processed
                    best_success = buf[2].item() / global_processed
                    format_rate = buf[3].item() / global_processed
                    if bar is not None:
                        bar.n = global_processed
                        bar.refresh()
                    print(f"[Global] ✅ processed={global_processed} | "
                          f"avg_success={avg_success:.2%} | best_success={best_success:.2%} | "
                          f"format={format_rate:.2%}")
            log_batch_stats(i // batch_size, batch_stats, rank)
            if rank == 0:
                print("累计统计:")
                log_batch_stats(i // batch_size, static, rank)

        overall_stats[dataset_name] = static
        
        torch.distributed.barrier()
        merge_results_from_all_ranks(save_path, cfg.eval.seed, world_size,cfg.eval.get('save_head', False))
        
        if bar is not None:
            bar.close()
    if rank == 0:
        with open(os.path.join(cfg.eval.save_path, f"overall_static_{cfg.eval.seed}.json"), "w") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=4)
    destroy_process_group()

if __name__ == "__main__":
    eval_main()