#!/usr/bin/env python3
"""
数据集格式检查脚本
检查现有数据集的格式，分析字段分布和类型
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict, Counter

# 添加e2c目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
e2c_dir = os.path.dirname(current_dir)
sys.path.insert(0, e2c_dir)

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")

def analyze_dataset_structure(data: List[Dict], dataset_name: str = "unknown") -> Dict[str, Any]:
    """
    分析数据集结构
    
    Args:
        data: 数据集列表
        dataset_name: 数据集名称
    
    Returns:
        分析结果字典
    """
    if not data:
        return {"error": "Empty dataset"}
    
    # 基本统计
    total_samples = len(data)
    
    # 字段分析
    field_counter = Counter()
    field_types = defaultdict(set)
    sample_types = []
    
    # 问题类型检测
    question_fields = set()
    answer_fields = set()
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            sample_types.append(f"non_dict_{type(item).__name__}")
            continue
        
        sample_types.append("dict")
        
        # 统计字段
        for field, value in item.items():
            field_counter[field] += 1
            field_types[field].add(type(value).__name__)
            
            # 检测问题字段
            if field.lower() in ['question', 'prompt', 'query', 'problem']:
                question_fields.add(field)
            
            # 检测答案字段
            if field.lower() in ['answer', 'response', 'solution', 'label']:
                answer_fields.add(field)
    
    # 分析问题内容
    question_analysis = analyze_questions(data, question_fields)
    
    # 分析答案内容
    answer_analysis = analyze_answers(data, answer_fields)
    
    return {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "sample_types": dict(Counter(sample_types)),
        "field_distribution": dict(field_counter),
        "field_types": {k: list(v) for k, v in field_types.items()},
        "question_fields": list(question_fields),
        "answer_fields": list(answer_fields),
        "question_analysis": question_analysis,
        "answer_analysis": answer_analysis,
        "format_compliance": check_format_compliance(data, question_fields, answer_fields)
    }

def analyze_questions(data: List[Dict], question_fields: Set[str]) -> Dict[str, Any]:
    """分析问题内容"""
    if not question_fields:
        return {"error": "No question fields found"}
    
    questions = []
    for item in data:
        if isinstance(item, dict):
            for field in question_fields:
                if field in item and item[field]:
                    questions.append(str(item[field]))
                    break
    
    if not questions:
        return {"error": "No questions found"}
    
    # 选择题检测
    choice_indicators = [
        "A)", "B)", "C)", "D)", "E)", "F)",
        "A.", "B.", "C.", "D.", "E.", "F.",
        "A、", "B、", "C、", "D、", "E、", "F、",
        "A．", "B．", "C．", "D．", "E．", "F．",
        "Choose", "Select", "Which", "What is the correct",
        "选择", "选出", "哪个", "哪项", "正确答案"
    ]
    
    multiple_choice_count = 0
    for question in questions:
        question_lower = question.lower()
        if any(indicator.lower() in question_lower for indicator in choice_indicators):
            multiple_choice_count += 1
    
    # 问题长度分析
    question_lengths = [len(q) for q in questions]
    
    return {
        "total_questions": len(questions),
        "multiple_choice_count": multiple_choice_count,
        "open_ended_count": len(questions) - multiple_choice_count,
        "multiple_choice_ratio": multiple_choice_count / len(questions) if questions else 0,
        "avg_question_length": sum(question_lengths) / len(question_lengths) if question_lengths else 0,
        "min_question_length": min(question_lengths) if question_lengths else 0,
        "max_question_length": max(question_lengths) if question_lengths else 0
    }

def analyze_answers(data: List[Dict], answer_fields: Set[str]) -> Dict[str, Any]:
    """分析答案内容"""
    if not answer_fields:
        return {"error": "No answer fields found"}
    
    answers = []
    for item in data:
        if isinstance(item, dict):
            for field in answer_fields:
                if field in item and item[field]:
                    answers.append(str(item[field]))
                    break
    
    if not answers:
        return {"error": "No answers found"}
    
    # 答案类型分析
    single_letter_answers = 0
    numeric_answers = 0
    text_answers = 0
    
    for answer in answers:
        answer_clean = answer.strip().upper()
        if answer_clean in ["A", "B", "C", "D", "E", "F"]:
            single_letter_answers += 1
        elif answer_clean.isdigit():
            numeric_answers += 1
        else:
            text_answers += 1
    
    # 答案长度分析
    answer_lengths = [len(a) for a in answers]
    
    return {
        "total_answers": len(answers),
        "single_letter_answers": single_letter_answers,
        "numeric_answers": numeric_answers,
        "text_answers": text_answers,
        "single_letter_ratio": single_letter_answers / len(answers) if answers else 0,
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
        "min_answer_length": min(answer_lengths) if answer_lengths else 0,
        "max_answer_length": max(answer_lengths) if answer_lengths else 0
    }

def check_format_compliance(data: List[Dict], question_fields: Set[str], answer_fields: Set[str]) -> Dict[str, Any]:
    """检查格式合规性"""
    compliance = {
        "has_question_field": len(question_fields) > 0,
        "has_answer_field": len(answer_fields) > 0,
        "all_items_dict": all(isinstance(item, dict) for item in data),
        "consistent_question_field": False,
        "consistent_answer_field": False,
        "ready_for_standardization": False
    }
    
    if not data:
        return compliance
    
    # 检查字段一致性
    if question_fields:
        # 检查是否所有样本都有问题字段
        question_field = list(question_fields)[0]
        compliance["consistent_question_field"] = all(
            question_field in item and item[question_field] 
            for item in data if isinstance(item, dict)
        )
    
    if answer_fields:
        # 检查是否所有样本都有答案字段
        answer_field = list(answer_fields)[0]
        compliance["consistent_answer_field"] = all(
            answer_field in item 
            for item in data if isinstance(item, dict)
        )
    
    # 检查是否准备好标准化
    compliance["ready_for_standardization"] = (
        compliance["has_question_field"] and 
        compliance["all_items_dict"] and
        compliance["consistent_question_field"]
    )
    
    return compliance

def load_dataset_file(file_path: str) -> List[Dict]:
    """加载数据集文件"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_path.suffix == '.parquet':
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library required for parquet files")
        dataset = Dataset.from_parquet(str(file_path))
        data = dataset.to_list()
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return data

def print_analysis_report(analysis: Dict[str, Any]):
    """打印分析报告"""
    print(f"\n{'='*60}")
    print(f"📊 数据集分析报告: {analysis['dataset_name']}")
    print(f"{'='*60}")
    
    # 基本统计
    print(f"\n📈 基本统计:")
    print(f"   总样本数: {analysis['total_samples']}")
    print(f"   样本类型: {analysis['sample_types']}")
    
    # 字段分析
    print(f"\n🏷️  字段分析:")
    print(f"   字段分布: {analysis['field_distribution']}")
    print(f"   字段类型: {analysis['field_types']}")
    
    # 问题分析
    if "error" not in analysis['question_analysis']:
        qa = analysis['question_analysis']
        print(f"\n❓ 问题分析:")
        print(f"   总问题数: {qa['total_questions']}")
        print(f"   选择题: {qa['multiple_choice_count']} ({qa['multiple_choice_ratio']*100:.1f}%)")
        print(f"   解答题: {qa['open_ended_count']} ({qa['open_ended_count']/qa['total_questions']*100:.1f}%)")
        print(f"   平均长度: {qa['avg_question_length']:.1f} 字符")
        print(f"   长度范围: {qa['min_question_length']} - {qa['max_question_length']} 字符")
    else:
        print(f"\n❓ 问题分析: {analysis['question_analysis']['error']}")
    
    # 答案分析
    if "error" not in analysis['answer_analysis']:
        aa = analysis['answer_analysis']
        print(f"\n💡 答案分析:")
        print(f"   总答案数: {aa['total_answers']}")
        print(f"   单字母答案: {aa['single_letter_answers']} ({aa['single_letter_ratio']*100:.1f}%)")
        print(f"   数字答案: {aa['numeric_answers']}")
        print(f"   文本答案: {aa['text_answers']}")
        print(f"   平均长度: {aa['avg_answer_length']:.1f} 字符")
        print(f"   长度范围: {aa['min_answer_length']} - {aa['max_answer_length']} 字符")
    else:
        print(f"\n💡 答案分析: {analysis['answer_analysis']['error']}")
    
    # 格式合规性
    print(f"\n✅ 格式合规性:")
    fc = analysis['format_compliance']
    print(f"   有问题字段: {'✅' if fc['has_question_field'] else '❌'}")
    print(f"   有答案字段: {'✅' if fc['has_answer_field'] else '❌'}")
    print(f"   所有项目为字典: {'✅' if fc['all_items_dict'] else '❌'}")
    print(f"   问题字段一致: {'✅' if fc['consistent_question_field'] else '❌'}")
    print(f"   答案字段一致: {'✅' if fc['consistent_answer_field'] else '❌'}")
    print(f"   可标准化: {'✅' if fc['ready_for_standardization'] else '❌'}")
    
    # 建议
    print(f"\n💡 建议:")
    if not fc['ready_for_standardization']:
        print("   ⚠️  数据集需要预处理才能使用标准化格式")
        if not fc['has_question_field']:
            print("   - 缺少问题字段")
        if not fc['all_items_dict']:
            print("   - 存在非字典格式的样本")
        if not fc['consistent_question_field']:
            print("   - 问题字段不一致")
    else:
        print("   ✅ 数据集可以直接使用预处理工具标准化")

def check_single_file(file_path: str):
    """检查单个文件"""
    try:
        print(f"🔍 检查文件: {file_path}")
        data = load_dataset_file(file_path)
        analysis = analyze_dataset_structure(data, Path(file_path).stem)
        print_analysis_report(analysis)
    except Exception as e:
        print(f"❌ 检查文件时出错: {e}")

def check_directory(directory: str, pattern: str = "*.parquet"):
    """检查目录中的所有文件"""
    directory = Path(directory)
    if not directory.exists():
        print(f"❌ 目录不存在: {directory}")
        return
    
    files = list(directory.glob(pattern))
    if not files:
        print(f"⚠️  在 {directory} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"📁 找到 {len(files)} 个文件")
    
    for file_path in files:
        check_single_file(str(file_path))
        print("\n" + "-"*60)

def main():
    parser = argparse.ArgumentParser(description="数据集格式检查工具")
    parser.add_argument("--input", "-i", required=True, help="输入文件或目录路径")
    parser.add_argument("--pattern", "-p", default="*.parquet", help="文件匹配模式 (默认: *.parquet)")
    parser.add_argument("--output", "-o", help="输出JSON报告文件路径")
    
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # 单文件检查
        check_single_file(args.input)
    elif os.path.isdir(args.input):
        # 目录批量检查
        check_directory(args.input, args.pattern)
    else:
        print(f"❌ 输入路径不存在: {args.input}")

if __name__ == "__main__":
    main()
