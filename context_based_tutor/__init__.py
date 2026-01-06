"""
Context-Based Tutor Module

A LangGraph-based intelligent tutoring system that adapts responses based on student's knowledge state.
Given a student's knowledge mastery state and a question, the system:
1. Decomposes the question into steps and maps each step to knowledge points
2. Compares required knowledge points with student's mastered knowledge points
3. Either provides direct answers (if student has mastered all required KPs) or guided tutoring (if gaps exist)
"""

from .run import run_context_based_tutor, run_context_based_tutor_async

__all__ = ['run_context_based_tutor', 'run_context_based_tutor_async']

