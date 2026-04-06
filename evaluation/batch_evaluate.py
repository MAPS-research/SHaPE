#!/usr/bin/env python3
"""
批量评估脚本：并行处理多个JSON文件
"""

import os
import sys
import asyncio
import glob
from pathlib import Path
from typing import List

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from pedagogical_evaluation import (
    PedagogicalEvaluator,
    process_all_results_async
)


async def batch_evaluate_json_files(
    input_dir: str,
    output_dir: str,
    evaluator_model: str = "gpt-5-mini",
    evaluator_provider: str = "openai",
    max_concurrent_files: int = 3,
    max_concurrent_evaluations: int = 10,
    student_state_dir: str = None
):
    """
    批量评估多个JSON文件
    
    Args:
        input_dir: 输入目录（包含JSON文件）
        output_dir: 输出目录
        evaluator_model: 评估器模型名称
        evaluator_provider: 评估器提供商
        max_concurrent_files: 最大并发处理的文件数
        max_concurrent_evaluations: 每个文件内的最大并发评估数
        student_state_dir: 学生状态CSV目录（可选）
    """
    print("=" * 80)
    print("📊 批量教学对话评估程序")
    print("=" * 80)

    if student_state_dir:
        print(f"⚠️  已提供 --student-state-dir: {student_state_dir}")
        print("   当前评估逻辑未使用该参数，将忽略并继续。")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有JSON文件（排除备份文件）
    json_pattern = os.path.join(input_dir, "*.json")
    json_files = [f for f in glob.glob(json_pattern) if not f.endswith('.backup')]
    
    if not json_files:
        print(f"❌ 在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"\n📁 找到 {len(json_files)} 个JSON文件")
    for i, f in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    # 初始化评估器
    print(f"\n🔧 初始化评估器...")
    evaluator = PedagogicalEvaluator(
        evaluator_model=evaluator_model,
        evaluator_provider=evaluator_provider
    )
    
    # 创建信号量控制文件级并发
    file_semaphore = asyncio.Semaphore(max_concurrent_files)
    
    async def process_single_file(json_file: str, file_index: int):
        """处理单个JSON文件"""
        async with file_semaphore:
            filename = os.path.basename(json_file)
            print(f"\n{'='*80}")
            print(f"[{file_index}/{len(json_files)}] 处理文件: {filename}")
            print(f"{'='*80}")
            
            # 生成输出文件名
            input_basename = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{input_basename}_evaluated.json")
            
            try:
                await process_all_results_async(
                    json_file=json_file,
                    evaluator=evaluator,
                    output_file=output_file,
                    max_concurrent=max_concurrent_evaluations
                )
                print(f"✅ [{file_index}/{len(json_files)}] 文件处理完成: {filename}")
                return True
            except Exception as e:
                print(f"❌ [{file_index}/{len(json_files)}] 文件处理失败: {filename}")
                print(f"   错误: {e}")
                return False
    
    # 创建所有文件处理任务
    tasks = [
        process_single_file(json_file, i + 1)
        for i, json_file in enumerate(json_files)
    ]
    
    # 并发执行所有任务
    print(f"\n🚀 开始批量处理（文件级并发: {max_concurrent_files}）...")
    start_time = asyncio.get_event_loop().time()
    
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time
    
    # 统计结果
    success_count = sum(1 for r in results if r)
    fail_count = len(results) - success_count
    
    print(f"\n{'='*80}")
    print("📊 批量处理完成")
    print(f"{'='*80}")
    print(f"  总文件数: {len(json_files)}")
    print(f"  成功: {success_count}")
    print(f"  失败: {fail_count}")
    print(f"  总耗时: {total_time:.2f}s (约 {total_time/60:.2f} 分钟)")
    print(f"  输出目录: {output_dir}")
    print("")
    
    # 列出生成的文件
    output_files = glob.glob(os.path.join(output_dir, "*_evaluated.json"))
    if output_files:
        print(f"✅ 生成的文件 ({len(output_files)} 个):")
        for f in sorted(output_files):
            print(f"  - {os.path.basename(f)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量评估教学对话JSON文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 批量评估目录中的所有JSON文件
  python batch_evaluate.py \\
      --input-dir /path/to/json/files \\
      --output-dir /path/to/output
  
  # 指定评估器和并发数
  python batch_evaluate.py \\
      --input-dir /path/to/json/files \\
      --output-dir /path/to/output \\
      --evaluator-model gpt-4o \\
      --evaluator-provider openai \\
      --max-concurrent-files 3 \\
      --max-concurrent-evaluations 10
        """
    )
    
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入目录（包含JSON文件）')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--evaluator-model', type=str, default='gpt-5-mini',
                       help='评估器模型名称（默认: gpt-5-mini）')
    parser.add_argument('--evaluator-provider', type=str, default='openai',
                       choices=['openai', 'gemini'],
                       help='评估器提供商（默认: openai）')
    parser.add_argument('--max-concurrent-files', type=int, default=3,
                       help='最大并发处理的文件数（默认: 3）')
    parser.add_argument('--max-concurrent-evaluations', type=int, default=10,
                       help='每个文件内的最大并发评估数（默认: 10）')
    parser.add_argument('--student-state-dir', type=str, default=None,
                       help='学生状态CSV文件目录（可选）')
    
    args = parser.parse_args()
    
    # 运行异步批量处理
    asyncio.run(batch_evaluate_json_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        evaluator_model=args.evaluator_model,
        evaluator_provider=args.evaluator_provider,
        max_concurrent_files=args.max_concurrent_files,
        max_concurrent_evaluations=args.max_concurrent_evaluations,
        student_state_dir=args.student_state_dir
    ))


if __name__ == "__main__":
    main()

