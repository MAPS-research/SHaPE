"""
Utility functions for the testing pipeline.
"""

import json
import os
import re
import itertools
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime


def load_questions(json_file: str, n: int = None) -> List[Dict[str, Any]]:
    """
    Load test questions from JSON file.
    
    Supports both JSON Lines format (one JSON object per line)
    and regular JSON array format.
    
    Args:
        json_file: Path to JSON file
        n: Number of questions to load (None for all)
        
    Returns:
        List of question dictionaries
    """
    questions = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        # Try JSON Lines format first
        content = f.read()
        lines = content.strip().split('\n')
        
        if len(lines) > 1:
            # JSON Lines format
            for line in lines:
                if line.strip():
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        else:
            # Regular JSON format
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    questions = data
                elif isinstance(data, dict) and 'questions' in data:
                    questions = data['questions']
            except json.JSONDecodeError:
                pass
    
    if n is not None:
        questions = questions[:n]
    
    print(f"Loaded {len(questions)} questions")
    return questions


def load_attack_method(attack_method: str) -> str:
    """
    Load attack method prompt.
    
    Args:
        attack_method: Attack method name or file path
            Supported names: baseline, refusal_suppression, role_play_en, role_play_cn, style_ristrict
        
    Returns:
        Attack prefix text (empty string for baseline)
    """
    # Baseline mode
    if attack_method.lower() in ['baseline', 'none', '']:
        print("Using Baseline mode (no jailbreak attack)")
        return ""
    
    # Check if it's a file path
    if os.path.exists(attack_method):
        with open(attack_method, 'r', encoding='utf-8') as f:
            attack_prefix = f.read().strip()
        print(f"Loaded attack method from: {os.path.basename(attack_method)}")
        return attack_prefix
    
    # Check predefined methods
    predefined_methods = {
        'refusal_suppression': 'refusal_suppression.txt',
        'role_play_en': 'role_play_en.txt',
        'role_play_cn': 'role_play_cn.txt',
        'style_ristrict': 'style_ristrict.txt'
    }
    
    if attack_method.lower() in predefined_methods:
        attack_file = os.path.join(
            os.path.dirname(__file__),
            '..',
            'attack_methods',
            predefined_methods[attack_method.lower()]
        )
        if os.path.exists(attack_file):
            with open(attack_file, 'r', encoding='utf-8') as f:
                attack_prefix = f.read().strip()
            print(f"Loaded attack method: {attack_method}")
            return attack_prefix
    
    raise ValueError(f"Unknown attack method: {attack_method}")


def identify_model_provider(model_name: str) -> str:
    """
    Automatically identify API provider from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Provider name (openai, gemini, claude, together_ai, openrouter, or None)
    """
    model_lower = model_name.lower()
    
    # OpenAI
    if any(kw in model_lower for kw in ['gpt', 'o1', 'o3']):
        return 'openai'
    
    # Gemini
    if 'gemini' in model_lower:
        return 'gemini'
    
    # Claude
    if 'claude' in model_lower:
        return 'claude'
    
    # OpenRouter (Qwen3)
    if 'qwen3' in model_lower or model_lower.startswith('openrouter/'):
        return 'openrouter'
    
    # Together AI
    if any(kw in model_lower for kw in ['llama', 'mixtral', 'qwen']):
        return 'together_ai'
    
    return None


def sample_student_states(
    required_kps: List[str],
    knowledge_graph,
    n_samples: int = 2,
    random_seed: int = 42
) -> List[List[str]]:
    """
    Sample valid missing knowledge point combinations.
    
    Generates all valid combinations and samples a subset.
    A combination is valid if it doesn't violate prerequisite relationships.
    
    Args:
        required_kps: Knowledge points required for the question
        knowledge_graph: KnowledgeGraph instance for validation
        n_samples: Number of samples to return
        random_seed: Random seed for reproducibility
        
    Returns:
        List of missing KP lists
    """
    rng = random.Random(random_seed)
    
    # Generate all possible combinations
    all_combinations = []
    for r in range(len(required_kps) + 1):
        for combo in itertools.combinations(required_kps, r):
            all_combinations.append(list(combo))
    
    # Filter valid combinations
    valid_combinations = []
    for missing_kps in all_combinations:
        try:
            if knowledge_graph.is_valid_missing_combination(required_kps, missing_kps):
                valid_combinations.append(missing_kps)
        except Exception:
            valid_combinations.append(missing_kps)
    
    # Sample
    k = min(n_samples, len(valid_combinations))
    return rng.sample(valid_combinations, k=k) if valid_combinations else [[]]


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    
    Uses simple heuristic: CJK characters count as 1 token each,
    non-CJK words (whitespace-separated) count as 1 token each.
    
    Args:
        text: Input text
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    cjk_re = re.compile(r"[\u4e00-\u9fff]")
    cjk_count = len(cjk_re.findall(text))
    non_cjk = cjk_re.sub(" ", text)
    chunks = re.findall(r"\S+", non_cjk)
    
    return cjk_count + len(chunks)


def clean_unicode_string(text: str) -> str:
    """
    Clean ambiguous Unicode characters from string.
    """
    if not isinstance(text, str):
        return text
    
    # Remove control characters (keep newline, tab, carriage return)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # Normalize spaces
    text = re.sub(r'[\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)
    text = re.sub(r' +', ' ', text)
    
    return text


def clean_dict_for_json(data: Any) -> Any:
    """
    Recursively clean all string values in a dict/list.
    """
    if isinstance(data, dict):
        return {key: clean_dict_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_dict_for_json(item) for item in data]
    elif isinstance(data, str):
        return clean_unicode_string(data)
    else:
        return data


def generate_output_filename(
    pipeline_type: str,
    model_name: str,
    attack_method: str,
    num_questions: int
) -> str:
    """
    Generate standardized output filename.
    
    Args:
        pipeline_type: 'context' or 'adaptive'
        model_name: Model name
        attack_method: Attack method name
        num_questions: Number of questions tested
        
    Returns:
        Filename string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean model name
    model_clean = model_name.replace('/', '-').replace('_', '-').replace(' ', '-')
    
    # Clean attack method name
    if attack_method.lower() == 'baseline':
        attack_clean = 'baseline'
    else:
        attack_clean = os.path.basename(attack_method)
        attack_clean = attack_clean.replace('.txt', '').replace('.TXT', '')
    attack_clean = attack_clean.replace('/', '-').replace('_', '-').replace(' ', '-')
    
    return f'{pipeline_type}-{model_clean}-{attack_clean}-{num_questions}-{timestamp}.json'

