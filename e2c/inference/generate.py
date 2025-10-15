import os
import sys
import json
from typing import List
import tqdm
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
from e2c.util.dataset import save_as_dataset,load_dataset_from_exploration
import torch
# Add e2c directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
e2c_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, e2c_dir)

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("Warning: hydra-core not installed. Using basic config loading.")
    hydra = None

from e2c.util.dataset import max_token_dataset, max_batch_size, is_multi_choice, load_dataset_by_name
from e2c.util.model import load_model

# VLLM imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: VLLM not available. Install with: pip install vllm")

def ddp_setup():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

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
    special_tokens += ['<EXPLORATION>','</EXPLORATION>','<EXECUTION>','</EXECUTION>']
    keep_special_token_ids = {tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    
    # 解码之前，过滤掉不在保留列表中的特殊 token
    filtered_token_ids = [
        token_id for token_id in token_ids
        if token_id not in tokenizer.all_special_ids or token_id in keep_special_token_ids
    ]
    
    # 使用过滤后的 token_ids 进行解码
    decoded_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=False)
    return decoded_text

def generate_batch_vllm(
    llm_model,
    tokenizer,
    input_texts,
    max_new_tokens=512,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=1.0,
    enable_thinking=False,
    solution_prefix="",
):
    """
    Generate responses using VLLM backend
    """
    if not VLLM_AVAILABLE:
        raise ImportError("VLLM not available. Please install with: pip install vllm")
    
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=sample_num,
        stop=["</s>"] + ([stop_token] if stop_token else []),
    )
    
    # Generate
    outputs = llm_model.generate(input_texts, sampling_params)
    
    # Process outputs
    results = []
    for output in outputs:
        sample_results = []
        for choice in output.outputs:
            generated_text = choice.text
            sample_results.append(generated_text)
        results.append(sample_results)
    
    return results

def generate_batch(
    model,
    tokenizer,
    input_text,
    max_new_tokens,
    device="cuda",
    stop_token=None,
    sample_num=1,
    temperature=1.0,
    top_p=0.7,
):
    """
    批量生成文本
    
    Args:
        model: 模型
        tokenizer: tokenizer
        input_text: 输入文本列表
        max_new_tokens: 最大生成token数
        device: 设备
        stop_token: 停止token
        sample_num: 每个输入的采样数量
        temperature: 温度
        top_p: top_p采样
        
    Returns:
        生成的文本列表，形状为 [sample_num][batch_size]
    """
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
    
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_length=input_ids.input_ids.shape[1] + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=sample_num,
            return_dict_in_generate=True,
            output_scores=False,  # 生成时不需要scores
            eos_token_id=[tokenizer.eos_token_id] if stop_token is None else [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids(stop_token)[0],
            ],
        )
        sequences = outputs.sequences[:, input_ids.input_ids.shape[1]:]  # only new tokens
        B = input_ids.input_ids.shape[0]  # batch size
        total_size = B * sample_num
        assert sequences.shape[0] == total_size

        # 初始化按 sample 排列
        all_results = []

        for s in range(sample_num):
            # 每个 batch 的第 s 个 sample
            batch_tokens = sequences[s::sample_num]  # (B, L)
            
            # 解码为文本
            batch_texts = []
            for i in range(B):
                token_ids = batch_tokens[i].cpu().tolist()
                text = decode_with_selected_special_tokens(
                    tokenizer, 
                    [], 
                    token_ids
                )
                batch_texts.append(text)
            
            all_results.append(batch_texts)
    
    return all_results

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

def merge_generations_from_all_ranks(save_path, seed, world_size):
    """合并所有rank的生成结果"""
    if torch.distributed.get_rank() != 0:
        return
        
    all_generations = []
    
    for rank in range(world_size):
        gen_file = os.path.join(save_path, f"generations_{seed}_rank{rank}.json")
        
        if os.path.exists(gen_file):
            with open(gen_file, "r") as f:
                generations = json.load(f)
            all_generations.extend(generations)
    
    # 保存合并后的结果
    if all_generations:
        with open(os.path.join(save_path, f"generations_{seed}_merged.json"), "w") as f:
            json.dump(all_generations, f, ensure_ascii=False, indent=4)
        
        print(f"\n✅ 生成合并完成: 总共 {len(all_generations)} 个样本")

def check_resume(save_path, seed, rank):
    """检查是否可以恢复之前的生成"""
    gen_file = os.path.join(save_path, f"generations_{seed}_rank{rank}.json")
    if os.path.exists(gen_file):
        with open(gen_file, "r") as f:
            generations = json.load(f)
        return generations
    return []

@hydra.main(version_base=None, config_path="../../e2c/config", config_name="generate")
def generate_main(cfg):
    # Check if using VLLM backend
    if cfg.model.type == "vllm":
        # VLLM uses single process but can utilize multiple GPUs via tensor parallelism
        device = "cuda"
        rank = 0  # VLLM uses single process
        world_size = 1  # VLLM uses single process
        torch.manual_seed(cfg.generation.seed)
        
        # Load VLLM model
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM not available. Please install with: pip install vllm")
        
        print(f"Loading VLLM model from: {cfg.model.model_path}")
        vllm_config = cfg.model.get('vllm', {})
        
        # Auto-detect number of GPUs for tensor parallelism
        num_gpus = torch.cuda.device_count()
        configured_tp_size = vllm_config.get('tensor_parallel_size', -1)
        
        if configured_tp_size == -1:
            # Auto-detect: use all available GPUs
            tensor_parallel_size = num_gpus
        else:
            # Use configured value
            tensor_parallel_size = min(configured_tp_size, num_gpus)
        
        print(f"Detected {num_gpus} GPUs, using tensor_parallel_size={tensor_parallel_size}")
        
        try:
            # Build VLLM configuration with performance optimizations
            vllm_kwargs = {
                'model': cfg.model.model_path,
                'tensor_parallel_size': tensor_parallel_size,
                'gpu_memory_utilization': vllm_config.get('gpu_memory_utilization', 0.85),
                'max_model_len': vllm_config.get('max_model_len', 8192),
                'dtype': vllm_config.get('dtype', 'bfloat16'),
                'trust_remote_code': vllm_config.get('trust_remote_code', True),
            }
            
            # Add performance optimization parameters
            if 'max_num_seqs' in vllm_config:
                vllm_kwargs['max_num_seqs'] = vllm_config['max_num_seqs']
            if 'max_num_batched_tokens' in vllm_config:
                vllm_kwargs['max_num_batched_tokens'] = vllm_config['max_num_batched_tokens']
            if 'enforce_eager' in vllm_config:
                vllm_kwargs['enforce_eager'] = vllm_config['enforce_eager']
            if 'enable_chunked_prefill' in vllm_config:
                vllm_kwargs['enable_chunked_prefill'] = vllm_config['enable_chunked_prefill']
            if 'enable_prefix_caching' in vllm_config:
                vllm_kwargs['enable_prefix_caching'] = vllm_config['enable_prefix_caching']
            if 'disable_log_stats' in vllm_config:
                vllm_kwargs['disable_log_stats'] = vllm_config['disable_log_stats']
            
            print(f"🚀 Loading VLLM with performance optimizations:")
            print(f"   - Tensor Parallel Size: {tensor_parallel_size}")
            print(f"   - GPU Memory Utilization: {vllm_kwargs['gpu_memory_utilization']}")
            print(f"   - Max Model Length: {vllm_kwargs['max_model_len']}")
            print(f"   - Max Num Seqs: {vllm_kwargs.get('max_num_seqs', 'default')}")
            print(f"   - Max Batched Tokens: {vllm_kwargs.get('max_num_batched_tokens', 'default')}")
            print(f"   - CUDA Graphs: {not vllm_kwargs.get('enforce_eager', True)}")
            
            model = LLM(**vllm_kwargs)
            print(f"✅ VLLM model loaded successfully with {tensor_parallel_size} GPUs")
        except Exception as e:
            print(f"❌ Failed to load VLLM model: {e}")
            raise
        
        # For VLLM, we don't need a separate tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
            print("✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            raise
    else:
        # Traditional DDP setup for other backends
        ddp_setup()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = f"cuda:{rank}"
        torch.manual_seed(cfg.generation.seed + rank)  # 不同rank使用不同的随机种子
        
        print(f"Loading traditional model from: {cfg.model.model_path}")
        try:
            model, tokenizer = load_model(cfg.model, device)
            print("✅ Traditional model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load traditional model: {e}")
            raise
        
        # 使用DDP包装模型
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[rank])
            print("✅ Model wrapped with DDP successfully")
        except Exception as e:
            print(f"❌ Failed to wrap model with DDP: {e}")
            raise
    
    if 'all' in cfg.generation.dataset:
        cfg.generation.dataset = ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]
    if 'med' in cfg.generation.dataset:
        cfg.generation.dataset = ["clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine","anatomy","medqa","medmcqa"]
    if 'resume' in cfg.generation.dataset:
        cfg.generation.dataset = [os.path.join(cfg.generation.resume_dir,dataset_name,f"generations_{cfg.generation.seed}_merged.json") for dataset_name in ["gsm8k", "math", "aime24", "aime25", "amc23", "math500", "minerva", "olympiad_bench"]]
        cfg.generation.dataset = [dataset_name for dataset_name in cfg.generation.dataset if os.path.exists(dataset_name)]
    for dataset_name in cfg.generation.dataset:
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name.endswith(".json"):
            dataset, dataset_name = load_dataset_from_exploration(dataset_name)
        else:
            try:
                dataset, dataset_name = load_dataset_by_name(dataset_name)
                print(f"✅ Dataset loaded: {len(dataset)} samples")
            except Exception as e:
                print(f"❌ Failed to load dataset {dataset_name}: {e}")
                continue
        
        if cfg.generation.get('use_default', False):
            if dataset_name in is_multi_choice:
                cfg.generation.system_prompt = r"You are a medical expert. You will be given a medical question and several candidate answers. Please choose the best answer based on your medical knowledge. You must give your answer in the format: 'The correct answer is boxed{A,B,C or D}'."
                cfg.generation.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."
            else:
                cfg.generation.system_prompt = r""
                cfg.generation.question_suffix = r"\nPlease reasoning step-by-step.Provide the final answer in the boxed{}."
        
        if cfg.generation.batch_size == -1:
            batch_size = max_batch_size.get(dataset_name, 10) // cfg.generation.sample_num
        else:
            batch_size = cfg.generation.batch_size // cfg.generation.sample_num
            
        if cfg.generation.max_new_tokens == -1:
            max_tokens = max_token_dataset.get(dataset_name, 1000)
        else:
            max_tokens = cfg.generation.max_new_tokens    
        
        save_path = os.path.join(cfg.generation.save_path, dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        if cfg.generation.get('resume', False):
            generations = check_resume(save_path, cfg.generation.seed, rank)
            print(f"[Rank {rank}] Resuming generation for dataset {dataset_name} from {len(generations)} existing results.")
        else:
            generations = []
        
        if cfg.model.type == "vllm":
            # VLLM mode: process all data in single process
            subset = dataset
            subset_size = len(dataset)
            bar = tqdm.tqdm(total=len(dataset), desc=f"Generating {dataset_name}", position=0, leave=True)
            start_idx = len(generations)
        else:
            # DDP mode: use distributed dataloader
            subset, subset_size = get_distributed_dataloader(dataset, batch_size, rank, world_size)
            
            if rank == 0:
                bar = tqdm.tqdm(total=len(dataset), desc=f"Generating {dataset_name}", position=0, leave=True)
            else:
                bar = None
            
            # 初始化进度条
            local_processed = len(generations)
            init_buf = torch.tensor([local_processed], device=device, dtype=torch.int64)
            torch.distributed.all_reduce(init_buf, op=torch.distributed.ReduceOp.SUM)
            
            if rank == 0 and init_buf[0].item() > 0:
                bar.n = init_buf[0].item()
                bar.refresh()
                print(f"[Init Global] processed={init_buf[0].item()}")
            
            start_idx = len(generations)
        
        for i in range(start_idx, subset_size, batch_size):
            batch_data = subset[i:i + batch_size]
            
            # 数据集已经是标准化的list[dict]格式，无需额外验证
            # 每个item包含: question, answer, type 字段
            # 统一的prompt处理，都使用tokenizer
            system_prompt = []
            enable_thinking = cfg.generation.get("enable_thinking", False)
            if "4B-Final" in cfg.model.model_path:
                enable_thinking = True
            if cfg.generation.get("system_prompt", None) is not None:
                system_prompt = [{
                    "role": "system",
                    "content": cfg.generation.system_prompt
                }]
            questions = [q["question"] + cfg.generation.get('question_suffix', '') for q in batch_data]
            questions = [
                tokenizer.apply_chat_template(
                    system_prompt + [
                        {"role": "user", "content": q}
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=cfg.generation.get("enable_thinking", True)
                )
                for q in questions
            ]
            questions = [q + cfg.generation.get("solution_prefix", "") for q in questions]
            
            if 'prompt' in batch_data[0]:
                questions = [q['prompt'] for q in batch_data]
            
            # 生成
            if cfg.model.type == "vllm":
                # Use VLLM generation
                samples = generate_batch_vllm(
                    model, 
                    tokenizer,
                    questions,
                    max_new_tokens=max_tokens,
                    device=device,
                    stop_token=cfg.generation.get("stop_token", None),
                    sample_num=cfg.generation.sample_num,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                    solution_prefix=cfg.generation.get("solution_prefix", ''),
                )
            else:
                # Use traditional generation
                samples = generate_batch(
                    model.module, 
                    tokenizer, 
                    questions,
                    max_tokens,
                    device=device,
                    stop_token=cfg.generation.get("stop_token", None),
                    sample_num=cfg.generation.sample_num,
                    temperature=cfg.generation.temperature,
                    top_p=cfg.generation.top_p,
                )
            
            # 保存结果
            for batch_idx in range(len(batch_data)):
                if cfg.model.type == "vllm":
                    # VLLM returns results[batch_idx][sample_idx]
                    responses = samples[batch_idx] if batch_idx < len(samples) else []
                else:
                    # Traditional model returns results[sample_idx][batch_idx]
                    responses = [samples[s][batch_idx] for s in range(cfg.generation.sample_num)]
                
                generation_item = {
                    "question": batch_data[batch_idx]["question"],
                    "answer": batch_data[batch_idx].get("answer", ""),  # 可能没有answer字段
                    "prompt": questions[batch_idx],
                    "responses": responses
                }
                generations.append(generation_item)
            
            # 定期保存
            if cfg.model.type == "vllm":
                # VLLM: save directly to final file (incremental save)
                output_file = os.path.join(save_path, f"generations_{cfg.generation.seed}_merged.json")
                with open(output_file, "w") as f:
                    json.dump(generations, f, ensure_ascii=False, indent=4)
                
                # Update progress bar
                if bar is not None:
                    bar.n = len(generations)
                    bar.refresh()
                print(f"[VLLM] ✅ processed={len(generations)}/{len(dataset)} - saved to {output_file}")
            else:
                # DDP: save per rank and merge later
                with open(os.path.join(save_path, f"generations_{cfg.generation.seed}_rank{rank}.json"), "w") as f:
                    json.dump(generations, f, ensure_ascii=False, indent=4)
                
                # 更新进度
                local_processed = len(generations)
                buf = torch.tensor([local_processed], device=device, dtype=torch.int64)
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
                
                if rank == 0:
                    global_processed = buf[0].item()
                    if bar is not None:
                        bar.n = global_processed
                        bar.refresh()
                    print(f"[Global] ✅ processed={global_processed}/{len(dataset)}")
        
        if bar is not None:
            bar.close()
        
        if cfg.model.type == "vllm":
            # VLLM: no need to merge, already saved to final file
            final_output_file = os.path.join(save_path, f"generations_{cfg.generation.seed}_merged.json")
            print(f"\n{'='*60}")
            print(f"[VLLM] Generation complete!")
            print(f"📁 Results saved to: {final_output_file}")
            print(f"📊 Total samples generated: {len(generations)}")
            print(f"🎯 Dataset: {dataset_name}")
            print(f"{'='*60}")
            if cfg.generation.save_as_dataset:
                print(f"💾 Saving as dataset...")
                save_as_dataset(generations, os.path.join(save_path, f"exploration_{cfg.generation.seed}.parquet"))
        else:
            # DDP: merge results from all ranks
            torch.distributed.barrier()
            merge_generations_from_all_ranks(save_path, cfg.generation.seed, world_size)
    
    # Only destroy process group if we're using DDP
    if cfg.model.type != "vllm":
        destroy_process_group()

if __name__ == "__main__":
    generate_main()
