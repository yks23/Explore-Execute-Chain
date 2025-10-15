#!/usr/bin/env python3
"""
数据集预处理脚本
统一数据集格式为：question, answer, type
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 添加e2c目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
e2c_dir = os.path.dirname(current_dir)
sys.path.insert(0, e2c_dir)

from e2c.util.dataset import detect_question_type, standardize_dataset_format
from datasets import Dataset

def preprocess_single_dataset(input_path: str, output_path: str = None, dataset_name: str = None) -> Dict[str, Any]:
    """
    预处理单个数据集文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（可选）
        dataset_name: 数据集名称（可选）
    
    Returns:
        预处理统计信息
    """
    print(f"🔄 处理数据集: {input_path}")
    
    # 加载原始数据集
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    elif input_path.endswith('.parquet'):
        raw_dataset = Dataset.from_parquet(input_path)
        raw_data = raw_dataset.to_list()
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")
    
    # 标准化格式
    standardized_data = standardize_dataset_format(raw_data, dataset_name)
    
    # 统计信息
    stats = {
        'total_samples': len(standardized_data),
        'multiple_choice': sum(1 for item in standardized_data if item['type'] == 'multiple_choice'),
        'open_ended': sum(1 for item in standardized_data if item['type'] == 'open_ended'),
        'with_answer': sum(1 for item in standardized_data if item['answer']),
        'without_answer': sum(1 for item in standardized_data if not item['answer'])
    }
    
    # 保存处理后的数据
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.json'):
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(standardized_data, f, ensure_ascii=False, indent=2)
        elif output_path.endswith('.parquet'):
            output_dataset = Dataset.from_list(standardized_data)
            output_dataset.to_parquet(output_path)
        
        print(f"✅ 已保存到: {output_path}")
    
    return stats

def preprocess_directory(input_dir: str, output_dir: str = None, file_pattern: str = "*.parquet"):
    """
    批量预处理目录中的所有数据集
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        file_pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找匹配的文件
    files = list(input_path.glob(file_pattern))
    if not files:
        print(f"⚠️  在 {input_dir} 中未找到匹配 {file_pattern} 的文件")
        return
    
    print(f"📁 找到 {len(files)} 个文件待处理")
    
    total_stats = {
        'total_files': len(files),
        'total_samples': 0,
        'total_multiple_choice': 0,
        'total_open_ended': 0,
        'total_with_answer': 0,
        'total_without_answer': 0
    }
    
    for file_path in files:
        try:
            # 确定输出路径
            if output_dir:
                output_path = Path(output_dir) / file_path.name
            else:
                output_path = None
            
            # 处理单个文件
            stats = preprocess_single_dataset(
                str(file_path), 
                str(output_path) if output_path else None,
                file_path.stem
            )
            
            # 累计统计
            total_stats['total_samples'] += stats['total_samples']
            total_stats['total_multiple_choice'] += stats['multiple_choice']
            total_stats['total_open_ended'] += stats['open_ended']
            total_stats['total_with_answer'] += stats['with_answer']
            total_stats['total_without_answer'] += stats['without_answer']
            
            print(f"   📊 {file_path.name}: {stats['total_samples']} 样本 "
                  f"({stats['multiple_choice']} 选择题, {stats['open_ended']} 解答题)")
            
        except Exception as e:
            print(f"❌ 处理 {file_path.name} 时出错: {e}")
    
    # 打印总体统计
    print("\n" + "="*50)
    print("📈 总体统计:")
    print(f"   处理文件数: {total_stats['total_files']}")
    print(f"   总样本数: {total_stats['total_samples']}")
    print(f"   选择题: {total_stats['total_multiple_choice']} ({total_stats['total_multiple_choice']/total_stats['total_samples']*100:.1f}%)")
    print(f"   解答题: {total_stats['total_open_ended']} ({total_stats['total_open_ended']/total_stats['total_samples']*100:.1f}%)")
    print(f"   有答案: {total_stats['total_with_answer']} ({total_stats['total_with_answer']/total_stats['total_samples']*100:.1f}%)")
    print(f"   无答案: {total_stats['total_without_answer']} ({total_stats['total_without_answer']/total_stats['total_samples']*100:.1f}%)")

def validate_dataset_format(dataset_path: str) -> bool:
    """
    验证数据集格式是否正确
    
    Args:
        dataset_path: 数据集文件路径
    
    Returns:
        是否格式正确
    """
    try:
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif dataset_path.endswith('.parquet'):
            dataset = Dataset.from_parquet(dataset_path)
            data = dataset.to_list()
        else:
            print(f"❌ 不支持的文件格式: {dataset_path}")
            return False
        
        if not isinstance(data, list):
            print(f"❌ 数据集应该是list格式，实际是: {type(data)}")
            return False
        
        required_fields = ['question', 'answer', 'type']
        valid_types = ['multiple_choice', 'open_ended']
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"❌ 第{i}项不是字典格式: {type(item)}")
                return False
            
            for field in required_fields:
                if field not in item:
                    print(f"❌ 第{i}项缺少字段 '{field}': {item}")
                    return False
            
            if item['type'] not in valid_types:
                print(f"❌ 第{i}项type字段值无效: {item['type']} (应该是 {valid_types})")
                return False
        
        print(f"✅ 数据集格式验证通过: {len(data)} 个样本")
        return True
        
    except Exception as e:
        print(f"❌ 验证数据集时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="数据集预处理工具")
    parser.add_argument("--input", "-i", required=True, help="输入文件或目录路径")
    parser.add_argument("--output", "-o", help="输出文件或目录路径")
    parser.add_argument("--validate", "-v", action="store_true", help="仅验证数据集格式")
    parser.add_argument("--pattern", "-p", default="*.parquet", help="文件匹配模式 (默认: *.parquet)")
    
    args = parser.parse_args()
    
    if args.validate:
        # 验证模式
        if os.path.isfile(args.input):
            validate_dataset_format(args.input)
        else:
            print("❌ 验证模式只支持单个文件")
    else:
        # 预处理模式
        if os.path.isfile(args.input):
            # 单文件处理
            preprocess_single_dataset(args.input, args.output)
        elif os.path.isdir(args.input):
            # 目录批量处理
            preprocess_directory(args.input, args.output, args.pattern)
        else:
            print(f"❌ 输入路径不存在: {args.input}")

if __name__ == "__main__":
    main()
