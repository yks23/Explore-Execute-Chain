import os
import json
import hydra
import torch
from typing import List
from util.reward import normalize_final_answer, extract_boxed_content
from util.visualization import plot_by_task_samples

def check_format(text, checklist):
    return all(key in text for key in checklist)

def multi_choice_evaluate(pred, gt):
    if 'boxed{' in pred:
        boxed_answer = pred.split("boxed{")[-1]
        boxed_answer = boxed_answer.split("text{")[-1]
        boxed_answer = boxed_answer.split("}")[0]
        if len(boxed_answer) == 1:
            if boxed_answer.isdigit():
                boxed_answer = {"1":"A","2":"B","3":"C","4":"D","5":"E"}.get(boxed_answer, boxed_answer)
            if boxed_answer == gt['answer_idx']:
                return True, boxed_answer
            else:
                return False, boxed_answer
    else:
        last_choice = ''
        for char in reversed(pred):
            if char in ['A','B','C','D','E']:
                last_choice = char
                break
            
        if last_choice:
            if last_choice == gt['answer_idx']:
                return True, last_choice
            else:
                return False, last_choice
        else:
            return None, "None"
    
    return False, "None"

def boxed_evaluate(pred, gt):
    gt, p_float = normalize_final_answer(gt)
    pre_answer = ""
    # find every boxed and compare with gt
    for pre_answer in extract_boxed_content(pred):
        pre_answer, q_float = normalize_final_answer(pre_answer)
        if pre_answer == gt or (q_float is not None and q_float == p_float):
            return True, pre_answer
    return False, pre_answer

def evaluate_generation_item(item, cfg):
    """
    评估单个生成项
    
    Args:
        item: 生成项，包含 question, answer, responses
        cfg: 配置
        
    Returns:
        评估结果字典
    """
    question = item["question"]
    gt_answer = item["answer"]
    responses = item['responses']
    sample_num = len(responses)
    
    per_sample_success = []
    per_sample_format = []
    per_sample_outputs = []
    failed_samples = []
    
    for s in range(sample_num):
        full_text = responses[s]
        boxed_count = full_text.count("boxed")
        
        if isinstance(gt_answer, dict):
            succ, pre_answer = multi_choice_evaluate(full_text, gt_answer)
            if succ == None:
                failed_samples.append(full_text)
                continue
            normalized_gt_answer = gt_answer['answer_idx']
        else:
            succ, pre_answer = boxed_evaluate(full_text, gt_answer)
            normalized_gt_answer, _ = normalize_final_answer(gt_answer)
        
        fmt = check_format(full_text, cfg.eval.checklist)
        per_sample_success.append(succ)
        per_sample_format.append(fmt)
        
        per_sample_outputs.append({
            "answer": pre_answer,
            "full_text": full_text,
            "success": succ,
            "format_success": fmt,
            "boxed_count": boxed_count
        })
    
    result = {
        "question": question,
        "normalized_gt_answer":normalized_gt_answer,
        "gt_answer": item['answer'],
        "samples": per_sample_outputs,
        "failed_samples": failed_samples,
        "avg_success": sum(per_sample_success) / len(per_sample_success) if per_sample_success else 0.25,
        "best_success": 1 if any(per_sample_success) else 0,
        "finish_rate": sum([1 for out in per_sample_outputs if out['boxed_count'] > 0]) / len(per_sample_outputs) if per_sample_outputs else 0,
        "finish_and_correct_rate": sum([1 for out in per_sample_outputs if out['boxed_count'] > 0 and out['success']]) / len(per_sample_outputs) if per_sample_outputs else 0,
    }
    
    return result

def find_generation_file(generation_path, dataset_name, seed):
    """
    自动查找生成结果文件
    
    Args:
        generation_path: 生成结果的基础路径
        dataset_name: 数据集名称
        seed: 随机种子
        
    Returns:
        找到的文件路径，如果没找到返回 None
    """
    # 可能的文件名模式
    possible_files = [
        f"generations_{seed}_merged.json",
        f"generations_{seed}.json",
    ]
    
    # 构建完整的搜索路径
    dataset_path = os.path.join(generation_path, dataset_name)
    
    # 按优先级搜索文件
    for filename in possible_files:
        full_path = os.path.join(dataset_path, filename)
        if os.path.exists(full_path):
            return full_path
    
    # 如果没找到，尝试在基础路径下直接搜索
    for filename in possible_files:
        full_path = os.path.join(generation_path, filename)
        if os.path.exists(full_path):
            return full_path
    
    return None
def group_by_question(generations):
    """把同一个question的所有的sample弄在一起"""
    grouped_generations = {}
    for generation in generations:
        question = generation["question"]
        if question not in grouped_generations:
            grouped_generations[question] = generation
        grouped_generations[question]['responses'].extend(generation['responses'])
    return list(grouped_generations.values())

def compute_statistics(results):
    """计算整体统计数据"""
    total = len(results)
    if total == 0:
        return {
            "avg_success": 0,
            "best_success": 0,
            "format_avg": 0,
            "all_number": 0,
            "finish_rate": 0,
            "finish_and_correct_rate": 0,
            "success_entropy": 0
        }
    
    avg_success = sum(r["avg_success"] for r in results) / total
    best_success = sum(r["best_success"] for r in results) / total
    format_avg = sum(r.get("samples", [{}])[0].get("format_success", 0) for r in results if r.get("samples")) / total
    
    # 计算完成率
    total_samples = sum(len(r["samples"]) for r in results)
    finish_count = sum(sum(1 for out in r['samples'] if out['boxed_count'] > 0) for r in results)
    finish_and_correct_count = sum(sum(1 for out in r['samples'] if out['boxed_count'] > 0 and out['success']) for r in results)
    
    finish_rate = finish_count / total_samples if total_samples > 0 else 0
    finish_and_correct_rate = finish_and_correct_count / finish_count if finish_count > 0 else 0
    
    # 计算成功熵
    entropies = []
    for result in results:
        avg_success_val = result.get("avg_success", 0)
        if avg_success_val > 0 and avg_success_val < 1:
            success_entropy = -avg_success_val * torch.log(torch.tensor(avg_success_val) + 1e-8) - (1 - avg_success_val) * torch.log(torch.tensor(1 - avg_success_val) + 1e-8)
            entropies.append(success_entropy.item())
    
    success_entropy = sum(entropies) / len(entropies) if entropies else 0
    
    return {
        "avg_success": avg_success,
        "best_success": best_success,
        "format_avg": format_avg,
        "all_number": total,
        "finish_rate": finish_rate,
        "finish_and_correct_rate": finish_and_correct_rate,
        "success_entropy": success_entropy
    }

@hydra.main(version_base=None, config_path="../config", config_name="eval_only")
def eval_only_main(cfg):
    print(f"\n{'='*50}")
    print("E2C Evaluation-Only")
    print(f"{'='*50}")
    
    overall_stats = {}
    
    # 处理数据集列表
    if 'all' in cfg.eval.dataset:
        cfg.eval.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    if 'med' in cfg.eval.dataset:
        cfg.eval.dataset = ["clinical_knowledge", "college_biology", "college_medicine", "medical_genetics", "professional_medicine", "anatomy", "medqa", "medmcqa"]
    
    for dataset_name in cfg.eval.dataset:
        print(f"\n{'='*50}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*50}")
        
        # 自动查找生成结果文件
        generation_path = find_generation_file(cfg.eval.generation_path, dataset_name, cfg.eval.seed)
        
        if not generation_path:
            print(f"❌ No generation file found for dataset {dataset_name} with seed {cfg.eval.seed}")
            print(f"   Searched in: {cfg.eval.generation_path}")
            continue
        
        print(f"📁 Found generation file: {generation_path}")
        
        with open(generation_path, "r") as f:
            generations = json.load(f)
        
        # 把同一个question的所有的sample弄在一起
        generations = group_by_question(generations)
        print(f"📊 Loaded {len(generations)} generations")
        
        # 评估所有生成结果
        results = []
        for item in generations:
            result = evaluate_generation_item(item, cfg)
            results.append(result)
        
        # 保存结果到 eval 文件夹
        eval_save_path = os.path.join(cfg.eval.save_path, dataset_name)
        os.makedirs(eval_save_path, exist_ok=True)
        
        # 保存详细结果
        result_file = os.path.join(eval_save_path, f"result_{cfg.eval.seed}_merged.json")
        with open(result_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        # 计算统计
        stats = compute_statistics(results)
        static_file = os.path.join(eval_save_path, f"static_{cfg.eval.seed}_merged.json")
        with open(static_file, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        
        print(f"💾 Results saved to: {result_file}")
        print(f"📊 Statistics saved to: {static_file}")
        
        print(f"\n📊 Statistics:")
        print(f"  - Average Success Rate: {stats['avg_success']:.2%}")
        print(f"  - Best Success Rate: {stats['best_success']:.2%}")
        print(f"  - Format Rate: {stats['format_avg']:.2%}")
        print(f"  - Finish Rate: {stats['finish_rate']:.2%}")
        print(f"  - Finish & Correct Rate: {stats['finish_and_correct_rate']:.2%}")
        print(f"  - Success Entropy: {stats['success_entropy']:.4f}")
        
        # 生成可视化
        # plot_by_task_samples(results, eval_save_path, 10)
        
        overall_stats[dataset_name] = stats
    
    # 保存总体统计
    with open(os.path.join(cfg.eval.save_path, f"overall_static_{cfg.eval.seed}.json"), "w") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=4)
    
    print(f"\n{'='*50}")
    print("Overall Statistics:")
    print(f"{'='*50}")
    for dataset_name, stats in overall_stats.items():
        print(f"{dataset_name}:")
        print(f"  - Avg Success: {stats['avg_success']:.2%}")
        print(f"  - Best Success: {stats['best_success']:.2%}")
    
    print(f"\n✅ Evaluation Complete!")

if __name__ == "__main__":
    eval_only_main()

