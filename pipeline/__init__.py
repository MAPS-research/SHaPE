"""
Testing Pipeline Module

Provides complete testing pipelines for:
1. Context-Based Tutor testing with jailbreak attacks
2. Adaptive Tutor testing with system-prompt approach

Features:
- Load test questions from JSON
- Generate student knowledge states
- Apply various jailbreak attack methods
- Run tests with multiple models
- Save results with full traceability
"""

from .utils import (
    load_questions,
    load_attack_method,
    identify_model_provider,
    sample_student_states
)

# Lazy imports for modules that require heavy dependencies (langgraph, etc.)
def run_context_tutor_test(*args, **kwargs):
    from .run_context_test import run_context_tutor_test as _fn
    return _fn(*args, **kwargs)

def run_adaptive_tutor_test(*args, **kwargs):
    from .run_adaptive_test import run_adaptive_tutor_test as _fn
    return _fn(*args, **kwargs)

__all__ = [
    'run_context_tutor_test',
    'run_adaptive_tutor_test',
    'load_questions',
    'load_attack_method',
    'identify_model_provider',
    'sample_student_states'
]

