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

from .run_context_test import run_context_tutor_test
from .run_adaptive_test import run_adaptive_tutor_test
from .utils import (
    load_questions,
    load_attack_method,
    identify_model_provider,
    sample_student_states
)

__all__ = [
    'run_context_tutor_test',
    'run_adaptive_tutor_test',
    'load_questions',
    'load_attack_method',
    'identify_model_provider',
    'sample_student_states'
]

