"""
Adaptive Tutor Testing Pipeline

Tests the Adaptive Tutor (system-prompt approach) with various jailbreak attacks.
"""

import os
import sys
import json
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from adaptive_tutor import run_adaptive_tutor_async
from adaptive_tutor.knowledge_graph import KnowledgeGraph
from pipeline.utils import (
    load_questions, 
    load_attack_method, 
    identify_model_provider,
    sample_student_states,
    estimate_tokens,
    clean_dict_for_json,
    generate_output_filename
)


async def run_adaptive_tutor_test(
    questions_file: str,
    adjacency_csv: str,
    client: Any,
    client_name: str,
    attack_method: str = "baseline",
    start_idx: int = 0,
    end_idx: int = None,
    n_samples_per_question: int = 2,
    use_dense_state: bool = True,
    max_concurrent: int = 10,
    random_seed: int = 42,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Run Adaptive Tutor test pipeline.
    
    Args:
        questions_file: Path to questions JSON file
        adjacency_csv: Path to knowledge graph adjacency matrix
        client: API client instance
        client_name: Name of the client/model
        attack_method: Attack method name or path
        start_idx: Starting question index
        end_idx: Ending question index (exclusive)
        n_samples_per_question: Number of student state samples per question
        use_dense_state: Whether to compute dense state including dependent KPs
        max_concurrent: Maximum concurrent API calls
        random_seed: Random seed for sampling
        output_dir: Output directory
        
    Returns:
        Complete results dictionary
    """
    print("=" * 80)
    print("Adaptive Tutor Testing Pipeline")
    print("=" * 80)
    
    # Load data
    attack_prefix = load_attack_method(attack_method)
    all_questions = load_questions(questions_file)
    knowledge_graph = KnowledgeGraph(adjacency_csv)
    
    # Select question range
    if end_idx is None:
        end_idx = len(all_questions)
    questions = all_questions[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} ({len(questions)} total)")
    
    # Generate test configurations
    test_configs = []
    for idx, question in enumerate(questions):
        question_id = f"q_{start_idx + idx}"
        required_kps = question.get('step', [])
        
        if not required_kps:
            continue
        
        # Sample valid missing KP combinations
        missing_samples = sample_student_states(
            required_kps, 
            knowledge_graph, 
            n_samples_per_question,
            random_seed + idx
        )
        
        for missing_kps in missing_samples:
            test_configs.append({
                "question_id": question_id,
                "question_index": start_idx + idx,
                "required_kps": required_kps,
                "missing_kps": missing_kps,
                "question_data": question
            })
    
    print(f"Generated {len(test_configs)} test configurations")
    
    # Run tests
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_test(config: Dict) -> Dict:
        async with semaphore:
            question_id = config["question_id"]
            missing_kps = config["missing_kps"]
            question = config["question_data"]
            
            print(f"Testing {question_id} with {len(missing_kps)} missing KPs...")
            
            start_time = time.time()
            
            try:
                result = await run_adaptive_tutor_async(
                    question=question.get('problem', ''),
                    missing_kps=missing_kps,
                    client=client,
                    adjacency_csv_path=adjacency_csv,
                    attack_prefix=attack_prefix,
                    use_dense_state=use_dense_state
                )
                
                end_time = time.time()
                
                return {
                    "question_id": question_id,
                    "question_index": config["question_index"],
                    "model": client_name,
                    "problem": question.get('problem', ''),
                    "expected_answer": question.get('answer', ''),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": missing_kps,
                    "model_answer": result.get("answer", ""),
                    "processing_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": bool(result.get("answer")) and not result.get("answer", "").startswith("Error"),
                    "token_usage": {
                        "prompt_tokens": estimate_tokens(result.get("system_prompt", "")) + estimate_tokens(result.get("user_prompt", "")),
                        "completion_tokens": estimate_tokens(result.get("answer", "")),
                        "estimated": True
                    },
                    "system_prompt": result.get("system_prompt", ""),
                    "user_prompt": result.get("user_prompt", ""),
                    "known_knowledge": result.get("known_knowledge", []),
                    "missing_knowledge": result.get("missing_knowledge", [])
                }
                
            except Exception as e:
                end_time = time.time()
                return {
                    "question_id": question_id,
                    "question_index": config["question_index"],
                    "model": client_name,
                    "problem": question.get('problem', ''),
                    "expected_answer": question.get('answer', ''),
                    "knowledge_points": config["required_kps"],
                    "input_missing_kps": missing_kps,
                    "model_answer": f"Error: {str(e)}",
                    "processing_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)
                }
    
    # Execute all tests concurrently
    tasks = [run_single_test(config) for config in test_configs]
    all_results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r.get('success', False))
    
    print(f"\nResults: {successful_tests}/{total_tests} successful ({100*successful_tests/total_tests:.1f}%)")
    
    # Build complete results
    complete_results = {
        'metadata': {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'pipeline': 'adaptive_tutor',
            'questions_file': questions_file,
            'start_index': start_idx,
            'end_index': end_idx,
            'total_questions': end_idx - start_idx,
            'attack_method': attack_method,
            'model': client_name,
            'use_dense_state': use_dense_state,
            'random_seed': random_seed,
            'n_samples_per_question': n_samples_per_question,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': round(100*successful_tests/total_tests, 2) if total_tests > 0 else 0,
        },
        'shared_config': {
            'attack_prefix': attack_prefix,
        },
        'results': all_results
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    filename = generate_output_filename('adaptive', client_name, attack_method, end_idx - start_idx)
    output_path = os.path.join(output_dir, filename)
    
    cleaned_results = clean_dict_for_json(complete_results)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    return complete_results


def main():
    """Command-line interface for adaptive tutor testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Tutor Testing Pipeline')
    
    parser.add_argument('--questions', type=str, required=True, help='Path to questions JSON file')
    parser.add_argument('--adjacency', type=str, required=True, help='Path to adjacency matrix CSV')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model to use')
    parser.add_argument('--attack', type=str, default='baseline', help='Attack method')
    parser.add_argument('--start', type=int, default=0, help='Start question index')
    parser.add_argument('--end', type=int, default=None, help='End question index')
    parser.add_argument('--samples', type=int, default=2, help='Samples per question')
    parser.add_argument('--no-dense-state', action='store_true', help='Disable dense state computation')
    parser.add_argument('--concurrent', type=int, default=10, help='Max concurrent calls')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    # Create client based on model
    provider = identify_model_provider(args.model)
    
    if provider == 'openai':
        from clients import OpenAIAnswerer
        client = OpenAIAnswerer(model=args.model)
    elif provider == 'gemini':
        from clients import GeminiAnswerer
        client = GeminiAnswerer(model=args.model)
    elif provider == 'claude':
        from clients import ClaudeAnswerer
        client = ClaudeAnswerer(model=args.model)
    else:
        raise ValueError(f"Could not identify provider for model: {args.model}")
    
    # Run test
    asyncio.run(run_adaptive_tutor_test(
        questions_file=args.questions,
        adjacency_csv=args.adjacency,
        client=client,
        client_name=args.model,
        attack_method=args.attack,
        start_idx=args.start,
        end_idx=args.end,
        n_samples_per_question=args.samples,
        use_dense_state=not args.no_dense_state,
        max_concurrent=args.concurrent,
        random_seed=args.seed,
        output_dir=args.output
    ))


if __name__ == "__main__":
    main()

