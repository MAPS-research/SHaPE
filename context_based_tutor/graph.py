"""
LangGraph workflow definition for the Context-Based Tutor.
Builds and compiles the state machine for intelligent tutoring.
"""

from typing import Dict, Any
from functools import partial
from langgraph.graph import StateGraph, END

from .state import TutorState
from .nodes import (
    node_decompose_and_map,
    node_compare_mastery,
    node_direct_answer,
    node_tutoring_answer,
)


def build_graph(llm: Any = None):
    """
    Build and compile the LangGraph workflow for tutoring.
    
    Workflow:
        Entry -> decompose_and_map -> compare_mastery -> (direct_answer | tutoring_answer) -> End
        
    The routing decision after compare_mastery is based on whether missing_kps is empty:
    - Empty missing_kps: student has all required knowledge -> direct answer
    - Non-empty missing_kps: knowledge gaps exist -> guided tutoring
    
    Args:
        llm: Optional custom LLM instance. If None, uses default OpenAI.
        
    Returns:
        Compiled LangGraph application
    """
    graph = StateGraph(TutorState)

    # Use partial to bind llm parameter to node functions
    graph.add_node("decompose_and_map", partial(node_decompose_and_map, llm=llm))
    graph.add_node("compare_mastery", partial(node_compare_mastery, llm=llm))
    graph.add_node("direct_answer", partial(node_direct_answer, llm=llm))
    graph.add_node("tutoring_answer", partial(node_tutoring_answer, llm=llm))

    graph.set_entry_point("decompose_and_map")

    def route_after_decompose(state: Dict) -> str:
        """Route based on whether fallback was used during decomposition."""
        return "fallback" if state.get("fallback_used") else "normal"

    graph.add_conditional_edges(
        "decompose_and_map",
        route_after_decompose,
        {
            "normal": "compare_mastery",
            "fallback": END,
        },
    )

    def route(state: Dict) -> str:
        """Route based on whether knowledge gaps exist."""
        return "need_tutoring" if state.get("missing_kps") else "direct"

    graph.add_conditional_edges(
        "compare_mastery",
        route,
        {
            "direct": "direct_answer",
            "need_tutoring": "tutoring_answer",
        },
    )
    
    graph.add_edge("direct_answer", END)
    graph.add_edge("tutoring_answer", END)

    return graph.compile()

