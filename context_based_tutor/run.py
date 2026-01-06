"""
Main entry points for the Context-Based Tutor module.
Provides both sync and async interfaces for running the tutoring workflow.
"""

import os
from typing import Dict, List, Set, Any

from .utils import load_knowledge_graph
from .graph import build_graph
from .adapters import adapt_llm


# Default path to adjacency matrix (relative to this file)
DEFAULT_ADJ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "adjacency_matrix_knowledge_graph.csv")


def _get_all_ancestors(missing_kps: List[str], kp_vocab: List[str], adj_matrix: List[List[int]]) -> Set[str]:
    """
    Recursively get all ancestor (prerequisite) knowledge points.
    
    Args:
        missing_kps: Initial list of missing knowledge points
        kp_vocab: Full list of knowledge point names
        adj_matrix: Adjacency matrix where adj_matrix[i][j] = 1 means KP[i] depends on KP[j]
        
    Returns:
        Set of all missing KPs including their ancestors
    """
    ancestors = set(missing_kps)
    to_check = set(missing_kps)

    while to_check:
        current = to_check.pop()
        if current in kp_vocab:
            idx = kp_vocab.index(current)
            # Find prerequisites (row idx in adjacency matrix)
            for j, dependency in enumerate(adj_matrix[idx]):
                if dependency == 1:
                    ancestor = kp_vocab[j]
                    if ancestor not in ancestors:
                        ancestors.add(ancestor)
                        to_check.add(ancestor)

    return ancestors


def run_context_based_tutor(
    question: str,
    missing_kps: List[str],
    adjacency_csv_path: str = None,
    llm: Any = None,
    debug: bool = False
) -> Dict:
    """
    Run the context-based tutor (synchronous version).
    
    Args:
        question: The student's question text
        missing_kps: List of knowledge points the student is missing
        adjacency_csv_path: Path to knowledge graph CSV. If None, uses default.
        llm: Optional custom LLM instance
        debug: If True, print debug information
        
    Returns:
        Dict containing:
            - final_answer: The tutor's response
            - trace: Debug information including steps, required_kps, missing_kps, etc.
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH
    
    kp_vocab, adj_matrix = load_knowledge_graph(adjacency_csv_path)

    # Calculate missing KPs including all ancestors
    missing_with_ancestors = _get_all_ancestors(missing_kps, kp_vocab, adj_matrix.tolist())

    # Mastered KPs = all KPs - missing KPs and their ancestors
    mastered_kps = [kp for kp in kp_vocab if kp not in missing_with_ancestors]

    init_state = {
        "question": question.strip(),
        "kp_vocab": kp_vocab,
        "mastered_kps": mastered_kps,
        "extra": {
            "missing_kps": list(missing_with_ancestors),
            "specified_missing_kps": missing_kps
        },
    }

    llm = adapt_llm(llm)
    app = build_graph(llm=llm)
    out = app.invoke(init_state)

    if debug:
        import json
        print("== steps ==")
        print(json.dumps(out.get("steps", []), ensure_ascii=False, indent=2))
        print("== required_kps ==")
        print(out.get("required_kps"))
        print("== missing_kps ==")
        print(out.get("missing_kps"))
        print("== specified_missing_kps ==")
        print(missing_kps)
        print("== direct_answer ==")
        print(out.get("direct_answer"))
        print("== tutoring_answer ==")
        print(out.get("tutoring_answer"))

    final_answer = out.get("direct_answer") or out.get("tutoring_answer") or ""
    return {
        "final_answer": final_answer,
        "trace": {
            "steps": out.get("steps", []),
            "required_kps": out.get("required_kps", []),
            "missing_kps": out.get("missing_kps", []),
            "specified_missing_kps": missing_kps,
            "fallback_used": out.get("fallback_used", False),
        }
    }


async def run_context_based_tutor_async(
    question: str,
    missing_kps: List[str],
    adjacency_csv_path: str = None,
    llm: Any = None,
    debug: bool = False
) -> Dict:
    """
    Run the context-based tutor (asynchronous version with true concurrency).
    
    Args:
        question: The student's question text
        missing_kps: List of knowledge points the student is missing
        adjacency_csv_path: Path to knowledge graph CSV. If None, uses default.
        llm: Optional custom LLM instance
        debug: If True, print debug information
        
    Returns:
        Dict containing:
            - final_answer: The tutor's response
            - trace: Debug information including steps, required_kps, missing_kps, etc.
    """
    if adjacency_csv_path is None:
        adjacency_csv_path = DEFAULT_ADJ_PATH
    
    kp_vocab, adj_matrix = load_knowledge_graph(adjacency_csv_path)

    # Calculate missing KPs including all ancestors
    missing_with_ancestors = _get_all_ancestors(missing_kps, kp_vocab, adj_matrix.tolist())

    # Mastered KPs = all KPs - missing KPs and their ancestors
    mastered_kps = [kp for kp in kp_vocab if kp not in missing_with_ancestors]

    init_state = {
        "question": question.strip(),
        "kp_vocab": kp_vocab,
        "mastered_kps": mastered_kps,
        "extra": {
            "missing_kps": list(missing_with_ancestors),
            "specified_missing_kps": missing_kps
        },
    }

    llm = adapt_llm(llm)
    app = build_graph(llm=llm)
    
    # Use ainvoke() for true async concurrency
    out = await app.ainvoke(init_state)

    final_answer = out.get("direct_answer") or out.get("tutoring_answer") or ""
    return {
        "final_answer": final_answer,
        "trace": {
            "steps": out.get("steps", []),
            "required_kps": out.get("required_kps", []),
            "missing_kps": out.get("missing_kps", []),
            "specified_missing_kps": missing_kps,
            "fallback_used": out.get("fallback_used", False),
        }
    }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    question = "Given the characteristic polynomial of matrix A, how to compute eigenvalues and eigenvectors for the 2x2 case?"
    missing_kps = ["Eigenvalues and Eigenvectors"]
    
    print("Running context-based tutor...")
    result = run_context_based_tutor(question, missing_kps, debug=True)
    print("\n=== Final Answer ===")
    print(result["final_answer"])

