import os
import json
import hydra
import torch
from typing import List
from torch.distributed import init_process_group, destroy_process_group
from util.reward import normalize_final_answer, extract_boxed_content
from util.visualization import plot_by_task_samples
from util.dataset import is_multi_choice, load_dataset_by_name
from util.model import EntropyCalculator, load_model

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def decode_with_selected_special_tokens(tokenizer, special_tokens, token_ids):
    """
    解码 token 序列，仅保留指定的特殊 token，移除其他特殊 token。
    """
    special_tokens += ['<THINKING>','</THINKING>','<EXPLORATION>','</EXPLORATION>','<EXECUTION>','</EXECUTION>']
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    
    decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
    return decoded_text

def decode_predictions(tokenizer, tokens, special_tokens):
    pred_texts = [
        decode_with_selected_special_tokens(tokenizer, special_tokens, ids) for ids in tokens
    ]
    answers = [txt[-200:] for txt in pred_texts]
    return pred_texts, answers

def check_format(text, checklist):
    return all(key in text for key in checklist)

def multi_choice_evaluate(pred, gt):
    if 'boxed{' in pred:
        boxed_answer = pred.split("boxed{")[-1]
        boxed_answer = boxed_answer.split("text{")[-1]
        boxed_answer = boxed_answer.split("}")[0]
        if len(boxed_answer)==1:
            if boxed_answer.isdigit():
                boxed_answer = {"1":"A","2":"B","3":"C","4":"D","5":"E"}.get(boxed_answer,boxed_answer)
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

def evaluate_batch(questions, gt_answers, samples, cfg, tokenizer):
    K = len(questions)
    sample_num = len(samples)
    entropy_calculator = EntropyCalculator(tokenizer)
    all_pred_texts = []
    all_answers = []
    
    for tokens, _, _ in samples:
        pred_texts, answers = decode_predictions(tokenizer, tokens, cfg.eval.head + cfg.eval.tail)
        all_pred_texts.append(pred_texts)
        all_answers.append(answers)
    
    entropy_stats = {}
    if samples:
        all_tokens = []
        all_entropies = []
        all_confidences = []
        for tokens, entropies, confidences in samples:
            all_tokens.extend(tokens)
            all_entropies.extend(entropies)
            all_confidences.extend(confidences)
        
        for start_token, end_token in zip(cfg.eval.head, cfg.eval.tail):
            stats = entropy_calculator.calculate_batch_entropy_stats(
                (all_tokens, all_entropies, all_confidences), start_token, end_token
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
            
            if isinstance(gt_answers[i], dict):
                succ, pre_answer = multi_choice_evaluate(full_text, gt_answers[i])
                if succ == None:
                    failed_samples.append(full_text)
                    continue
            else:
                succ, pre_answer = boxed_evaluate(full_text, gt_answers[i])

            fmt = check_format(full_text, cfg.eval.checklist)
            per_sample_success.append(succ)
            per_sample_format.append(fmt)
            
            per_sample_outputs.append({
                "answer": pre_answer,
                "full_text": full_text,
                "success": succ,
                "format_success": fmt,
                "static": entropy_calculator.calculate_entropy_stats((tokens, entropies, confidences), cfg.eval.head[0], cfg.eval.tail[0]) if cfg.eval.need_static else None,
                "boxed_count": boxed_count
            })

        success_avg += sum(per_sample_success) / len(per_sample_success) if per_sample_success else 0.25
        format_avg += sum(per_sample_format) / len(per_sample_format) if per_sample_format else 0.25
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
        "entropy_stats": entropy_stats
    }
    
    return results, batch_stats

def merge_results_from_all_ranks(save_path, seed, world_size, save_head=False):
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
            
            if static["all_number"] > 0:
                weight = static["all_number"]
                all_static["avg_success"] = (all_static["avg_success"] * total_samples + static["avg_success"] * weight) / (total_samples + weight)
                all_static["best_success"] = (all_static["best_success"] * total_samples + static["best_success"] * weight) / (total_samples + weight)
                all_static["format_avg"] = (all_static["format_avg"] * total_samples + static["format_avg"] * weight) / (total_samples + weight)
                total_samples += weight
                
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
    
    entropies = []
    for result in all_results:
        avg_success = result.get("avg_success", 0)
        success_entropy = -avg_success*torch.log(torch.tensor(avg_success) + 1e-8) - (1 - avg_success)*torch.log(torch.tensor(1 - avg_success) + 1e-8)
        entropies.append(success_entropy.item())
        finish_count += sum([1 for out in result['samples'] if out['boxed_count']>0])
        finish_and_correct_count += sum([1 for out in result['samples'] if out['boxed_count']>0 and out['success']])
    
    all_static['finish_rate'] = finish_count / total_samples if total_samples > 0 else 0
    all_static['finish_and_correct_rate'] = finish_and_correct_count / finish_count if finish_count > 0 else 0
    all_static['success_entropy'] = sum(entropies) / len(entropies) if len(entropies) > 0 else 0
    
    if save_head:
        q2a = {}
        for result in all_results:
            question = result['question']
            answer = result['samples'][0]['full_text']
            q2a[question] = answer
        with open(os.path.join(save_path, f"q2a.json"), "w") as f:
            json.dump(q2a, f, ensure_ascii=False, indent=4)
    
    if all_results:
        with open(os.path.join(save_path, f"result_{seed}_merged.json"), "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(save_path, f"static_{seed}_merged.json"), "w") as f:
            json.dump(all_static, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 合并完成: 总共 {len(all_results)} 个样本")
        print(f"📊 最终统计: 平均成功率={all_static['avg_success']:.2%}, "
              f"最好成功率={all_static['best_success']:.2%}, "
              f"格式率={all_static['format_avg']:.2%}")
    
    plot_by_task_samples(all_results, save_path, 10)

@hydra.main(version_base=None, config_path="../config", config_name="eval")
def evaluate_main(cfg):
    ddp_setup()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = f"cuda:{rank}"
    
    # 加载tokenizer用于解码
    _, tokenizer = load_model(cfg.model, device)
    
    overall_stats = {}
    
    if 'all' in cfg.eval.dataset:
        cfg.eval.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    if 'med' in cfg.eval.dataset:
        cfg.eval.dataset = ["clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","anatomy","medqa","medmcda"]
    
    for dataset_name in cfg.eval.dataset:
        dataset, dataset_name = load_dataset_by_name(dataset_name)
        
        save_path = os.path.join(cfg.eval.save_path, dataset_name)
        
        # 读取生成的结果
        generation_file = os.path.join(save_path, f"generation_{cfg.eval.seed}_rank{rank}.json")
        if not os.path.exists(generation_file):
            print(f"[Rank {rank}] Generation file not found: {generation_file}")
            continue
        
        with open(generation_file, "r") as f:
            generated_data = json.load(f)
        
        print(f"[Rank {rank}] Evaluating {len(generated_data)} samples for dataset {dataset_name}")
        
        results = []
        static = {"avg_success": 0, "best_success": 0, "format_avg": 0, "all_number": 0, "entropy_stats": {}}
        
        # 评测每个样本
        for data in generated_data:
            question = data["question"]
            gt_answer = data["gt_answer"]
            
            # 重建samples格式
            samples = []
            for sample in data["samples"]:
                tokens = sample["tokens"]
                entropies = sample["entropies"]
                confidences = sample["confidences"]
                samples.append((tokens, entropies, confidences))
            
            # 重组为按sample_num分组的格式
            sample_num = len(samples)
            reorganized_samples = []
            for s in range(sample_num):
                reorganized_samples.append(([samples[s][0]], [samples[s][1]], [samples[s][2]]))
            
            # 评测
            batch_results, batch_stats = evaluate_batch(
                [question], [gt_answer], reorganized_samples, cfg, tokenizer
            )
            
            results.extend(batch_results)
            
            # 更新统计信息
            if not static["all_number"]:
                static = batch_stats
            else:
                static["avg_success"] = (static["avg_success"] * static["all_number"] + batch_stats["avg_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
                static["best_success"] = (static["best_success"] * static["all_number"] + batch_stats["best_success"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
                static["format_avg"] = (static["format_avg"] * static["all_number"] + batch_stats["format_avg"] * batch_stats["all_number"]) / (static["all_number"] + batch_stats["all_number"])
                static["all_number"] += batch_stats["all_number"]
                
                # 合并熵统计
                for key, new_stats in batch_stats.get("entropy_stats", {}).items():
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
        
        # 保存评测结果
        with open(os.path.join(save_path, f"result_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        with open(os.path.join(save_path, f"static_{cfg.eval.seed}_rank{rank}.json"), "w") as f:
            json.dump(static, f, ensure_ascii=False, indent=4)
        
        print(f"[Rank {rank}] Evaluation completed for {dataset_name}")
        print(f"[Rank {rank}] Avg success: {static['avg_success']:.2%}, Best success: {static['best_success']:.2%}")
        
        overall_stats[dataset_name] = static
        
        torch.distributed.barrier()
        merge_results_from_all_ranks(save_path, cfg.eval.seed, world_size, cfg.eval.get('save_head', False))
    
    if rank == 0:
        with open(os.path.join(cfg.eval.save_path, f"overall_static_{cfg.eval.seed}.json"), "w") as f:
            json.dump(overall_stats, f, ensure_ascii=False, indent=4)
    
    destroy_process_group()

if __name__ == "__main__":
    evaluate_main()

