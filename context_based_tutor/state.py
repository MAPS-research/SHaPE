"""
State definitions for the Context-Based Tutor workflow.
Defines the TypedDict structure used throughout the LangGraph state machine.
"""

from typing import List, TypedDict, Optional, Dict, Any


class StepDict(TypedDict):
    """Structure for a single decomposition step"""
    description: str
    kps: List[str]


class TutorState(TypedDict, total=False):
    """
    Main state object for the tutor workflow.
    
    Attributes:
        question: The student's question text
        kp_vocab: List of all knowledge point candidates (aligned with adjacency matrix)
        steps: LLM-decomposed steps with mapped knowledge points
        required_kps: All required knowledge points for this question (deduplicated, order preserved)
        mastered_kps: Knowledge points the student has mastered
        missing_kps: Knowledge points the student lacks but are required for this question
        direct_answer: Direct solution text (when missing_kps is empty)
        tutoring_answer: Guided tutoring response text (when missing_kps exists)
        student_csv_path: Optional path to student state file
        extra: Container for additional information
    """
    question: str
    kp_vocab: List[str]
    steps: List[StepDict]
    required_kps: List[str]
    mastered_kps: List[str]
    missing_kps: List[str]
    direct_answer: Optional[str]
    tutoring_answer: Optional[str]
    student_csv_path: Optional[str]
    extra: Dict[str, Any]

