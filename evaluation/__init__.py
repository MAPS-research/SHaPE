"""
Evaluation Module

Provides tools for evaluating pedagogical quality of LLM tutoring responses:
- Safety: Does the model avoid giving direct answers when it shouldn't?
- Helpfulness: Does the model provide correct answers when appropriate?
- Pedagogy: Does the model guide students toward target knowledge points?
"""

from .pedagogical_evaluation import (
    PedagogicalEvaluator,
    StudentStateReader,
    evaluate_single_result_async,
    evaluate_single_result,
    process_all_results_async,
    process_all_results,
)

__all__ = [
    'PedagogicalEvaluator',
    'StudentStateReader',
    'evaluate_single_result_async',
    'evaluate_single_result',
    'process_all_results_async',
    'process_all_results',
]
